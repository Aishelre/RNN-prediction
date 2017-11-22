"""
** 실시간으로 데이터를 받아옴.
**
** client에서 list데이터를 보내고
**
** server에서 이 데이터를 seq_length 크기의 배치에 넣는다.
**
** 배치[0]의 데이터를 삭제하고 [9]에 데이터를 추가 하여
** TF로 예측을 한다. 
**
** 예측 결과인 int 값을 client로 넘겨준다.
** 이 값을 가지고 거래 여부를 결정한다.

★ server.py는 계속 실행 되기 때문에
★ 직접 종료해주어야 한다.

"""

import Pyro4
import tensorflow as tf
import numpy as np
import json


@Pyro4.expose
class TF(object):
    def __init__(self):
        with open("Save data/hyper parameters.txt", 'rt') as fp:
            param = json.loads(fp.readline())
        tf.set_random_seed(0)
        np.random.seed(0)
        # Hyper Parameters=
        self.seq_length = param['seq_length']
        self.data_dim = param['data_dim']
        self.hidden_dim = param['hidden_dim']
        self.output_dim = 2
        self.pkeep = param['pkeep']
        self.NLAYERS = param['NLAYERS']

        self.input_data = np.zeros((1, self.seq_length, self.data_dim), float)
        self.result = 0
        self.output_ = 0


    def wrapper(self, processed_data):
        processed_data = np.array([[processed_data]])
        self.input_data = np.delete(self.input_data, 0, axis=1)
        self.input_data = np.concatenate((self.input_data, processed_data), axis=1)

        self.tf_init()
        print("init 함수 종료")
        self.prediction()
        idx = int(np.argmax(self.output_))
        print("Argmax : ",idx)
        return idx

    def prediction(self):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state('Save data/')
            print(ckpt)
            self.saver = tf.train.Saver()

            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                return

            self.output_ = sess.run(self.output, {self.tf_x: self.input_data})
            # accuracy_ = sess.run(accuracy, {tf_x: test_input, tf_y: test_label})
            # print(test_input, '->', test_label)
            print("------------------------")
            print("* output : ", self.output_)
            # print('* test accuracy  : %.2f' % accuracy_)

        tf.reset_default_graph()


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


    # End Of Class


tf_ins = TF()
processed_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
tf_ins.wrapper(processed_data)

daemon = Pyro4.Daemon()
uri = daemon.register(TF)
print("Ready. uri = ", uri)
daemon.requestLoop()