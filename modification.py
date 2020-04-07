# modification.py - программа, которая преобразовывает аудиофайл,  
# используя систему распознавания голоса deepspeechv0.4.1.
# На вход принимает исходный аудиофайл и желаемую транскрипцию на английском языке.
# Выходом является аудиофайл с некоторыми погрешностями, который обнаруживается 
# системой deepspeech, как желаемая транскрипция.

import numpy as np
import tensorflow as tf
import argparse

from shutil import copyfile

import scipy.io.wavfile as wav

import sys
import os
import struct
import time

from collections import namedtuple

sys.path.append("DeepSpeech")

import pydub

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

# Токены, которые мы используем. Токен '-' особый, он соответствует эпсилон значению в CTC.
# Он не может встречаться в транскрипции

toks = " abcdefghijklmnopqrstuvwxyz'-"

def convert_mp3(one, length):
    import pydub
    wav.write("/tmp/load.wav", 16000,
              np.array(np.clip(np.round(one[0][:length[0]]),
                               -2**15, 2**15-1),dtype=np.int16))
    pydub.AudioSegment.from_wav("/tmp/load.wav").export("/tmp/saved.mp3")
    cur = pydub.AudioSegment.from_mp3("/tmp/saved.mp3")
    mp3e = np.array([struct.unpack("<h", cur.cur_data[i:i+2])[0] for i in range(0,len(cur.cur_data),2)])[np.newaxis,:length[0]]
    return mp3e
    

class Modify:
    def __init__(self, sess, loss_fn, phrase_length, maxlen,
                 learn_rate=10, iterations_num=5000, mem_size=1,
                 mp3=False, foreit=float('inf'), restore_path=None):
               
        ## Настроим процедуру modify

        ## Здесь создаётся tf граф, который мы используем, чтобы генерировать аудиофайл.
        
        self.sess = sess
        self.learn_rate = learn_rate
        self.iterations_num = iterations_num
        self.mem_size = mem_size
        self.phrase_length = phrase_length
        self.maxlen = maxlen
        self.mp3 = mp3


        # Создаём необходимые переменные Они имеют префикс qq, чтобы отличаться
        # от стандартных. Таким образом мы отличаем их от остальных



        self.delta = delta = tf.Variable(np.zeros((mem_size, maxlen), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((mem_size, maxlen), dtype=np.float32), name='qq_mask')
        self.maskcw = maskcw = tf.Variable(np.zeros((mem_size, phrase_length), dtype=np.float32), name='qq_maskcw')
        self.oring = oring = tf.Variable(np.zeros((mem_size, maxlen), dtype=np.float32), name='qq_oring')
        self.length = length = tf.Variable(np.zeros(mem_size, dtype=np.int32), name='qq_length')
        self.importance = tf.Variable(np.zeros((mem_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((mem_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_length = tf.Variable(np.zeros((mem_size), dtype=np.int32), name='qq_phrase_length')
        self.rescale = tf.Variable(np.zeros((mem_size,1), dtype=np.float32), name='qq_phrase_length')


        # Изначально привяжем  l_infty к 2000, увеличиваем константу, если она 
        # недостаточно велика для искажения нашего набора данных.


        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale

        # Мы устанавливаем новый вход для модели, чтобы получить дельту и маску,
        # которая позволяет применять определённым значениям константу 0 для 
        # последовательного заполнения длины.


        self.new_input = new_input = self.apply_delta*mask + oring
        

        # Добавляем шума, чтобы убедиться, что можно обрезать значения 
        # в 16-битные целые числа.

        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # Вводим конечное число, чтобы получить logits.
 
        self.logits = logits = get_logits(pass_in, length)

        # Здесь восстанавливаем график, чтобы сделать классификатор

        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        # Выбираем функцию потерь - СТС или CW. 
        # В нашем случае это CTC.

        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_length)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=length)

            # Небольшая оговорка: бесконечный штраф l2 означает, что мы не увеличиваем 
            # искажение l2. Код работает быстрее при небольшой величине искажения, а также
            # оставляет на единицу меньше параметр, который требует настройки

            if not np.isinf(foreit):
                loss = tf.reduce_mean((self.new_input-self.oring)**2,axis=1) + foreit*ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
            
        elif loss_fn == "CW": #  Введём предупреждение, что  modify() не поддерживает CW.
            raise NotImplemented("Сurrent version does not support implementation CW.")
        else:
            raise

        self.loss = loss
        self.ctcloss = ctcloss
        
        # Настроим AdamOptimizer для выполнения градиентного спуска.

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learn_rate)

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        #  Декодер logits нужен для того, чтобы просмотреть успешность выполнения программы 

        
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, length, merge_repeated=False, beam_width=100)

    def modify(self, audio, length, target, finetune=None):
        sess = self.sess

        # Объявление переменных.
        # Каждая из этих операций создаёт новый график tf.
        # Они создаются один раз. Работает только если вызывать modify() несколько раз


        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.oring.assign(np.array(audio)))
        sess.run(self.length.assign((np.array(length)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.maxlen)] for l in length])))
        sess.run(self.maskcw.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(length)-1)//320])))
        sess.run(self.target_phrase_length.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.mem_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.mem_size,1))))

        # Здесть мы отслеживаем лучшее решение, которое нашли

        final_deltas = [None]*self.mem_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # Здесь мы делаем много итераций градиентного спуска

        now = time.time()
        MAX = self.iterations_num
        for i in range(MAX):
            iteration = i
            now = time.time()

            # Выводим отладочную информацию, чтобы проверять состояние работы программы
            # Информация выводится каждые 5 итераций


            if i%5 == 0:
                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]
                if self.mp3:
                    mp3e = convert_mp3(new, length)
                    
                    mp3_out, mp3_logits = sess.run((self.decoded, self.logits),
                                                   {self.new_input: mp3e})
                    lst.append((mp3_out, mp3_logits))

                for out, logits in lst:
                    chars = out[0].values

                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]
                    
                    # Здесь мы печатаем строки, которые распознаются.

                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # Выводим argmax выравнивания
                    
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,length)]
                    print("\n".join(res2))


            if self.mp3:
                new = sess.run(self.new_input)
                mp3e = convert_mp3(new, length)
                feed_dict = {self.new_input: mp3e}
            else:
                feed_dict = {}

            d, el, cl, l, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss, self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    
            # Отчёт о прогрессе
            
            print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            for ii in range(self.mem_size):

                # Каждые 100 итераций проверяем, добились ли мы успеха
                # Если получилось (или это последняя эпоха), то мы должны 
                # записывать прогресс и уменьшать константу масштабирования 
                

                if (self.loss_fn == "CTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    
                    # Получаем текущую константу
                    
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):

                        # Если мы уже прошли порог, то нужно уменьшить
                        # порог до текущего точки, тем самым
                        # сэкономим немного времени 
                        
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0
                    # В противном случае, уменьшаем его на некоторую константу.
                    # Чем ближе это число к 1, тем лучше качества результата.
                    # Чем меньше, тем быстрее будет схождение на результат, но
                    # это будет плохо сказываться на качестве.

                    rescale[ii] *= .8

                    # Отрегулируем лучшее решение, найденное до сих пор 

                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d losscoef=%f dif=%f"%(ii,cl[ii], 2000*rescale[ii][0]))
                    print('Delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    sess.run(self.rescale.assign(rescale))

                    
        return final_deltas
    
    
def main():

    # Здесь мы обрабатываем модификацию. 

    # Здесь представлен шаблон
    # Как было сказанно раньше, поддерживается только функция потери СТС 
    # и модификация только одного файла за раз.

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file, at 16KHz")
    parser.add_argument('--out', type=str, nargs='+',
                        required=False,
                        help="Path for the modified file")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Transcription")
    parser.add_argument('--outprefix', type=str,
                        required=False,
                        help="Prefix of path for the modified file")
    parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file to use as a starting point")
    parser.add_argument('--lr', type=int,
                        required=False, default=100,
                        help="Learning rate")
    parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--foreit', type=float,
                        required=False, default=float('inf'),
                        help="Weight for foreit on loss function")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    parser.add_argument('--mp3', action="store_const", const=True,
                        required=False,
                        help="Generate MP3 compression resistant audiofiles")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()
    
    with tf.Session() as sess:
        finetune = []
        audios = []
        length = []

        if args.out is None:
            assert args.outprefix is not None
        else:
            assert args.outprefix is None
            assert len(args.input) == len(args.out)
        if args.finetune is not None and len(args.finetune):
            assert len(args.input) == len(args.finetune)
        
        # Загружаем данные, которые подаются на вход.

        for i in range(len(args.input)):
            fs, audio = wav.read(args.input[i])
            assert fs == 16000
            assert audio.dtype == np.int16
            print('source dB', 20*np.log10(np.max(np.abs(audio))))
            audios.append(list(audio))
            length.append(len(audio))

            if args.finetune is not None:
                finetune.append(list(wav.read(args.finetune[i])[1]))

        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

        phrase = args.target

        # Ставим класс modify и запускаем его.

        modify = Modify(sess, 'CTC', len(phrase), maxlen,
                        mem_size=len(audios),
                        mp3=args.mp3,
                        learn_rate=args.lr,
                        iterations_num=args.iterations,
                        foreit=args.foreit,
                        restore_path=args.restore_path)
        deltas = modify.modify(audios,
                               length,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               finetune)

        # Сохраняем результат в текущей папке.

        if args.mp3:
            convert_mp3(deltas, length)
            copyfile("/tmp/saved.mp3", args.out[0])
            print("Final distortion", np.max(np.abs(deltas[0][:length[0]]-audios[0][:length[0]])))
        else:
            for i in range(len(args.input)):
                if args.out is not None:
                    path = args.out[i]
                else:
                    path = args.outprefix+str(i)+".wav"
                wav.write(path, 16000,
                          np.array(np.clip(np.round(deltas[i][:length[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
                print("Final distortion", np.max(np.abs(deltas[i][:length[i]]-audios[i][:length[i]])))

main()
