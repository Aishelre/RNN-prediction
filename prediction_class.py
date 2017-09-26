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

class Prediction():
    def __init__(self):
        with open("hyper parameters.txt", 'rt') as fp:
            param = json.loads(fp.readline())
        tf.set_random_seed(0)
        np.random.seed(0)
        # Hyper Parameters=
        self.BATCH_SIZE = param['BATCH_SIZE']
        self.LR = param['LR']
        self.seq_length = param['seq_length']
        self.data_dim = param['data_dim']
        self.hidden_dim = param['hidden_dim']
        self.output_dim = param['output_dim']
        self.iterations = param['iterations']
        self.interval = param['interval']

        self.csv_file = glob.glob("* - processed.csv")
        print(self.csv_file)

        self.start()

    def start(self):
        # self.cnt_ ~ 같은 변수들 초기화
        for file in self.csv_file:
            self.tf_init()
            self.set_dataset(file)
            self.predict(file)

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

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y, logits=self.output)  # compute cost
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1), name='acc')[1]
        # It returns (acc, update_op), and create 2 local variables

    def set_dataset(self, filename):
        xy = np.loadtxt(filename, delimiter=',')  # , usecols=range(data_dim + 2))  # numpy.ndarray 타입
        self.data_dim = len(xy[0]) - 2

        self.cnt_num_label = [0 for _ in range(self.output_dim)]
        self.true2true = [0 for _ in range(self.output_dim)]
        self.cnt_precision = [0 for _ in range(self.output_dim)]

        x = xy[..., 2:]  # 시간, 현재 가격을 제외한 호가 항목
        y = xy[..., 1]  # 현재 가격*

        self.input = np.empty((1, self.seq_length, self.data_dim), float)
        self.label = np.empty((1, self.output_dim), int)

        # label을, 현재가 변화량이 +일 때 +1, -일 때 -1, 0일때 0으로 초기화
        print("label data generating - - -")
        for i in range(self.seq_length - 1, len(y) - self.interval):
            val = y[i + self.interval] - y[i]
            if val > 0:
                lb = np.array([0, 0, 1])  # +
                self.cnt_num_label[2] += 1
            elif val < 0:
                lb = np.array([1, 0, 0])  # -
                self.cnt_num_label[0] += 1
            else:
                lb = np.array([0, 1, 0])  # 0
                self.cnt_num_label[1] += 1
            self.label = np.vstack((self.label, lb))
        self.label = self.label[1:]  # np.empty에서 생성한 label[0]을 지운다.

        print("input data generating - - -")
        for i in range(0, len(y) - self.seq_length+1 - self.interval):
            input_ = np.array([x[i:i + self.seq_length]])
            self.input = np.vstack((self.input, input_))
        self.input = self.input[1:]

    def predict(self, filename):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state('Save data/')
            self.saver = tf.train.Saver()
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("## SAVED DATA LOAD ##")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("** tf.global_variables_initializer **")
                # sess.run(tf.global_variables_initializer())

            output_,accuracy_ = sess.run([self.output, self.accuracy], {self.tf_x: self.input, self.tf_y: self.label})

        print("------------------------")
        print("* result         : ", filename)
        print('* output         : ',output_)
        print('* test accuracy  : %.2f' % accuracy_)
        print("------------------------")

        label = (np.argmax(self.label, 1))
        out = (np.argmax(output_, 1))
        for idx in range(len(label)):
            if label[idx] == out[idx]:
                self.true2true[label[idx]] += 1
            self.cnt_precision[out[idx]] += 1
        print("Precision : 오른다고 예측했는데 실제로 오른 비율 (실제 / 예측 비율 (예측이 얼마나 맞는지)")
        print("label[0] precision : ", self.true2true[0] / self.cnt_precision[0])
        print("label[1] precision : ", self.true2true[1] / self.cnt_precision[1])
        print("label[2] precision : ", self.true2true[2] / self.cnt_precision[2])
        print("Recall : 실제로 오르는데 오른다고 예측한 비율")
        print("label[0] recall : ", self.true2true[0] / self.cnt_num_label[0])
        print("label[1] recall : ", self.true2true[1] / self.cnt_num_label[1])
        print("label[2] recall : ", self.true2true[2] / self.cnt_num_label[2])
        print("------------------------")
        tf.reset_default_graph()
        # -- End of Training --

print(" ============================== ")
print(" ====== PREDICTION MODEL ====== ")
print(" ============================== ")
tf_ins = Prediction()