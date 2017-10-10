"""
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import glob

'''
# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()
'''

def mod(a, b):
    if b == 0:
        result = 0
    else:
        result = a / b
    return result

class Prediction():
    def __init__(self):
        self.csv_file = glob.glob("* - processed.csv")
        print(self.csv_file)

        self.start()

    def start(self):
        # self.cnt_ ~ 같은 변수들 초기화
        for file in self.csv_file:
            with open("CKPT/result_N " + file, 'wt') as fp:
                fp.write("ckpt, acc, precision(-), precision(0), precision(+), recall(-), recall(0), recall(+)\n")
            with open("CKPT/result_P " + file, 'wt') as fp:
                fp.write("ckpt, acc, precision(-), precision(0), precision(+), recall(-), recall(0), recall(+)\n")

            ckpt_dir = glob.glob("CKPT/*Save data")
            print(ckpt_dir)
            for c in ckpt_dir:
                print("ckpt : ",c)
                with open(c+"/Hyper parameters.txt", 'rt') as fp:
                    param = json.loads(fp.readline())
                tf.set_random_seed(0)
                np.random.seed(0)
                # Hyper Parameters=
                self.interval = param['interval']
                self.seq_length = param['seq_length']
                self.data_dim = param['data_dim']
                self.hidden_dim = param['hidden_dim']
                self.output_dim = 2

                self.set_dataset(file)

                self.tf_init()
                if "P Save data" in c:
                    print("POSITIVE, ZERO prediction")
                    self.label = self.posi
                    self.predict_P(file, c)
                elif "N Save data" in c:
                    print("NEGATIVE, ZERO prediction")
                    self.label = self.nega
                    self.predict_N(file, c)

    def tf_init(self):
        self.tf_x = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim], name='tf_x')
        self.tf_y = tf.placeholder(tf.int32, [None, self.output_dim], name='tf_y')

        # RNN
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim)
        self.outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
            self.rnn_cell,  # cell you have chosen
            self.tf_x,  # input
            initial_state=None,  # the initial hidden state
            dtype=tf.float32,  # must given if set initial_state = None
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
            scope="rnn"
        )
        self.output = tf.layers.dense(self.outputs[:, -1, :], self.output_dim, name='dense_output')

        #self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y, logits=self.output)  # compute cost
        #self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1), name='acc')[1]
        # It returns (acc, update_op), and create 2 local variables

    def set_dataset(self, filename):
        json_filename = filename + " " + str(self.interval) + " " + str(self.seq_length) + " " + ".json"

        with open("DATA/seqed data/" + json_filename, 'rt') as fp:
            data = json.load(fp)
            self.cnt_num_label = data[0]
            self.input = np.array(data[1])
            self.label = np.array(data[2])
            self.train_size = len(self.label)
            fp.close()

        self.posi = np.delete(self.label, np.s_[0], axis=1)  # + binary

        for i in range(len(self.posi)):
            if (self.posi[i] == np.array([0, 0])).all():
                self.posi[i] = np.array([1, 0])


        ##########################
        self.nega = np.delete(self.label, np.s_[2], axis=1)  # + binary
        for i in range(len(self.nega)):
            if (self.nega[i] == np.array([0, 0])).all():
                self.nega[i] = np.array([0, 1])

    def predict_P(self, filename, ckpt_path):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state(ckpt_path+'/')
            print(ckpt)
            self.saver = tf.train.Saver()
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("## SAVED DATA LOAD ##")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("** tf.global_variables_initializer **")
                # sess.run(tf.global_variables_initializer())

            output_,accuracy_ = sess.run([self.output, self.accuracy], {self.tf_x: self.input, self.tf_y: self.label})

        print("------------------------")
        print("* result    : ", filename)
        print('* accuracy  : %.2f' % accuracy_)
        print("------------------------")

        label = np.argmax(self.label, 1)  # (n, 3) array의 각 행의 max값 index를 array로 반환.
        out = np.argmax(output_, 1)
        self.true2true = [0 for _ in range(self.output_dim)]
        self.cnt_precision = [0 for _ in range(self.output_dim)]
        for idx in range(len(label)):
            if label[idx] == out[idx]:
                self.true2true[label[idx]] += 1
            self.cnt_precision[out[idx]] += 1
        precision_0 = mod(self.true2true[0] , self.cnt_precision[0])
        precision_1 = mod(self.true2true[1] , self.cnt_precision[1])
        recall_0 = mod(self.true2true[0] , self.cnt_num_label[1] + self.cnt_num_label[0])
        recall_1 = mod(self.true2true[1] , self.cnt_num_label[2])
        print("Precision : 오른다고 예측했는데 실제로 오른 비율 (실제 / 예측 비율 (예측이 얼마나 맞는지)")
        print("label[0] precision : ", precision_0)
        print("label[+] precision : ", precision_1)
        print("Recall : 실제로 오르는데 오른다고 예측한 비율")
        print("label[0] recall : ", recall_0)
        print("label[+] recall : ", recall_1)
        print("------------------------")

        with open("CKPT/result_P "+filename, 'at') as fp:
            fp.write(str(ckpt_path) + ',')
            fp.write(str(accuracy_) + ',')
            fp.write(",")
            fp.write(str(precision_0) + ',' + str(precision_1) + ',')
            fp.write(",")
            fp.write(str(recall_0) + ',' + str(recall_1) + ',' + '\n')

        tf.reset_default_graph()
        # -- End of Prediction --
    def predict_N(self, filename, ckpt_path):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state(ckpt_path+'/')
            print(ckpt)
            self.saver = tf.train.Saver()
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("## SAVED DATA LOAD ##")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("** tf.global_variables_initializer **")
                # sess.run(tf.global_variables_initializer())

            output_,accuracy_ = sess.run([self.output, self.accuracy], {self.tf_x: self.input, self.tf_y: self.label})

        print("------------------------")
        print("* result    : ", filename)
        print('* accuracy  : %.2f' % accuracy_)
        print("------------------------")

        label = np.argmax(self.label, 1)  # (n, 3) array의 각 행의 max값 index를 array로 반환.
        out = np.argmax(output_, 1)
        self.true2true = [0 for _ in range(self.output_dim)]
        self.cnt_precision = [0 for _ in range(self.output_dim)]
        for idx in range(len(label)):
            if label[idx] == out[idx]:
                self.true2true[label[idx]] += 1
            self.cnt_precision[out[idx]] += 1
        precision_0 = mod(self.true2true[0] , self.cnt_precision[0])
        precision_1 = mod(self.true2true[1] , self.cnt_precision[1])
        recall_0 = mod(self.true2true[0] , self.cnt_num_label[0])
        recall_1 = mod(self.true2true[1] , self.cnt_num_label[1]+self.cnt_num_label[2])
        print("Precision : 오른다고 예측했는데 실제로 오른 비율 (실제 / 예측 비율 (예측이 얼마나 맞는지)")
        print("label[-] precision : ", precision_0)
        print("label[0] precision : ", precision_1)
        print("Recall : 실제로 오르는데 오른다고 예측한 비율")
        print("label[-] recall : ", recall_0)
        print("label[0] recall : ", recall_1)
        print("------------------------")

        with open("CKPT/result_N "+filename, 'at') as fp:
            fp.write(str(ckpt_path) + ',')
            fp.write(str(accuracy_) + ',')
            fp.write(str(precision_0) + ',' + str(precision_1) + ',')
            fp.write(",")
            fp.write(str(recall_0) + ',' + str(recall_1) + ',' + '\n')

        tf.reset_default_graph()
        # -- End of Prediction --

print(" ============================== ")
print(" ====== PREDICTION MODEL ====== ")
print(" ============================== ")
tf_ins = Prediction()
