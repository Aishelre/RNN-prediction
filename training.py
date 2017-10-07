"""
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import glob
import os

'''
# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()
'''

class Training():
    def __init__(self):
        self.path = "CKPT/Save data"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open("hyper parameters.txt", 'rt') as fp:
            param = json.loads(fp.readline())
            fp.close()
        with open(self.path+"/params.txt", 'wt') as fp:
            fp.write(str(param))
            fp.close()
        tf.set_random_seed(0)
        np.random.seed(0)
        # Hyper Parameters=
        self.denominator = param['denominator']
        self.BATCH_SIZE = param['BATCH_SIZE']
        self.LR = param['LR']
        self.seq_length = param['seq_length']
        self.data_dim = param['data_dim']
        self.hidden_dim = param['hidden_dim']
        self.output_dim = param['output_dim']
        self.iterations = param['iterations']
        self.interval = param['interval']
        print(self.hidden_dim)
        self.csv_file = glob.glob("./DATA/Trainable-file/* - processed.csv")
        print(self.csv_file)

        self.train()

    def train(self):
        for file in self.csv_file:
            self.set_dataset(file)
            self.tf_init()
            self.training(file)

    def tf_init(self):
        self.tf_x = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim], name='tf_x')
        self.tf_y = tf.placeholder(tf.float32, [None, self.output_dim], name='tf_y')

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
        self.accuracy = \
        tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1), name='acc')[1]
        # It returns (acc, update_op), and create 2 local variables

    def set_dataset(self, filename):
        xy = np.loadtxt(filename, delimiter=',')  # , usecols=range(data_dim + 2))  # numpy.ndarray 타입
        self.data_dim = len(xy[0]) - 2
        print("data_dim : ", self.data_dim)

        self.cnt_num_label = [0 for _ in range(self.output_dim)]
        self.cnt_label = [0 for _ in range(self.output_dim)]

        x = xy[..., 2:]  # 시간, 현재 가격을 제외한 호가 항목
        y = xy[..., 1]  # 현재 가격*

        self.input = np.empty((1, self.seq_length, self.data_dim), float)
        self.label = np.empty((1, self.output_dim), int)

        # len(input) = file_length - ( seq_length - 1 ) => range(0, file_length-seq_length+1)
        # len(label) = file_length - seq_length - interval => range(seq_length+interval-1, file_length)
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
        # print("LABEL : ", label)  # if(seq_length = 10) : len(y) = 12258 -> len(label) = 12248

        print("input data generating - - -")
        for i in range(0, len(y) - self.seq_length + 1):
            input_ = np.array([x[i:i + self.seq_length]])
            self.input = np.vstack((self.input, input_))
        self.input = self.input[1:]

        self.train_size = len(self.label)

    def training(self, filename):
        with tf.Session() as sess:
            print("class 분포[ -, 0, + ] : ", self.cnt_num_label)  # 전체 분포
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state(self.path+"/")
            print(ckpt)
            self.saver = tf.train.Saver()
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("## SAVED DATA LOAD ##")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("** tf.global_variables_initializer **")
                # sess.run(tf.global_variables_initializer())

            for step in range(self.iterations + 1):  # training
                batch_input = np.empty((1, self.seq_length, self.data_dim), float)
                batch_label = np.empty((1, self.output_dim), int)

                i = 1
                cnt_label = [0 for _ in range(self.output_dim)]
                while (i <= self.BATCH_SIZE):
                    idx = random.randint(0, self.train_size - 1)
                    input_ = np.array([self.input[idx]])
                    label_ = np.array(self.label[idx])

                    if label_.tolist() == [0, 0, 1]:
                        cnt_label[2] += 1
                    elif label_.tolist() == [0, 1, 0]:
                        if cnt_label[1] >= int(self.BATCH_SIZE / self.denominator):
                            continue
                        cnt_label[1] += 1
                    elif label_.tolist() == [1, 0, 0]:
                        cnt_label[0] += 1

                    batch_input = np.vstack((batch_input, input_))
                    batch_label = np.vstack((batch_label, label_))

                    i += 1
                print("cnt_label : ", cnt_label)
                batch_input = batch_input[1:]
                batch_label = batch_label[1:]

                _, loss_ = sess.run([self.train_op, self.loss], {self.tf_x: batch_input, self.tf_y: batch_label})

                if step % 50 == 0:
                    print("------------------------")
                    print("step             : %d" % step)
                    print('* train loss     : %.4f' % loss_)

            save_path = self.saver.save(sess, self.path+"/RNN-model.ckpt")

        print("------------------------")
        print(save_path)
        print("* result         : ", filename)
        print('* train loss     : %.4f' % loss_)
        print("------------------------")
        # -- End of Training --
        tf.reset_default_graph()


tf_ins = Training()