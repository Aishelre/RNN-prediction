"""
Request : trained model, 1초 데이터
Var : data, asset=0, cnt=0

1) processed csv파일을 readlines로 저장한다.
2) ‎한 줄씩 읽어 TF로 보낸다.
3) ‎return값이 2일 때 asset-=now, 0일 때 asset+=now. cnt를 증/감 시킨다.
4) ‎파일을 모두 읽으면 asset+=(cnt * last now)
"""

import tensorflow as tf
import numpy as np
import json



class TF(object):
    def __init__(self):
        with open("Save data/hyper parameters.txt", 'rt') as fp:
            param = json.loads(fp.readline())
        tf.set_random_seed(0)
        np.random.seed(0)
        # Hyper Parameters=
        self.interval = param['interval']
        self.seq_length = param['seq_length']
        self.data_dim = param['data_dim']
        self.hidden_dim = param['hidden_dim']
        self.output_dim = 2

        self.pkeep = param["pkeep"]
        self.NLAYERS = param['NLAYERS']

        self.input_data = np.zeros((1, self.seq_length, self.data_dim), float)
        self.result = 0
        self.output_ = 0

        self.flag = True

    def wrapper(self, processed_data):
        processed_data = np.array([[processed_data]])
        self.input_data = np.delete(self.input_data, 0, axis=1)
        self.input_data = np.concatenate((self.input_data, processed_data), axis=1)

        self.tf_init()
        output = self.prediction()

        return np.argmax(output, axis=1)[0]

    def prediction(self):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state('Save data/')
            self.saver = tf.train.Saver()
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)

            self.output_ = sess.run(self.output, {self.tf_x: self.input_data})
            print("* output : ", self.output_)
        tf.reset_default_graph()
        return self.output_

    def tf_init(self):
        self.tf_x = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim], name='tf_x')
        self.tf_y = tf.placeholder(tf.float32, [None, self.output_dim], name='tf_y')

        # How to properly apply dropout in RNNs: see README.md
        self.cells = [tf.contrib.rnn.GRUCell(num_units=self.hidden_dim) for _ in range(self.NLAYERS)]
        # "naive dropout" implementation
        self.dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.pkeep) for cell in self.cells]
        self.multicell = tf.contrib.rnn.MultiRNNCell(self.dropcells, state_is_tuple=False)
        self.multicell = tf.contrib.rnn.DropoutWrapper(self.multicell, output_keep_prob=self.pkeep)  # dropout for the softmax layer

        self.outputs, _ = tf.nn.dynamic_rnn(self.multicell, self.tf_x, dtype=tf.float32, initial_state=None, scope="rnn")
        self.output = tf.layers.dense(self.outputs[:, -1, :], self.output_dim, name='dense_output')

        #self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.tf_y, logits=self.output)  # compute cost
        #self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(self.output, axis=1), name='acc')[1]
        # It returns (acc, update_op), and create 2 local variables


filename = "09.27 043200 - processed.csv"
asset = 0
cnt = 0

tf_ins = TF()
tf_ins.wrapper([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

with open(filename, 'rt') as fp:
    data = fp.readlines()
    for line in data:
        line = line.strip().split(',')
        line = list(map(float, line))
        print(type(line[2]), line)
        output = tf_ins.wrapper(line[2:])
        print("argmax label : ", output)
        if output == 0 and cnt != 0:
            cnt -= 1
            asset += line[1]
        if output == 1:
            cnt += 1
            asset -= line[1]

        print("평가액 : ",asset + cnt*line[1])
        now = line[1]
        print("===========================================")
    #end of prediction
    if cnt > 0:
        print(cnt)
        asset += cnt * now

print("평가액 : ", asset)
