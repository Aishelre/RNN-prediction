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
        self.csv_file = glob.glob("DATA/Trainable-file/* - processed.csv")
        print(self.csv_file)

        self.cnt = 1
        self.hyper_param_init()

    def hyper_param_init(self):
        with open("hyper parameters.txt", 'rt') as fp:
            param = json.loads(fp.readline())
            fp.close()
        tf.set_random_seed(0)
        np.random.seed(0)
        # Hyper Parameters=
        self.denominator = param['denominator']
        self.BATCH_SIZE = param['BATCH_SIZE']
        self.LR = param['LR']
        self.seq_length = param['seq_length']
        self.hidden_dim = param['hidden_dim']
        self.iterations = param['iterations']
        self.interval = param['interval']
        self.data_dim = param['data_dim']
        self.output_dim = param['output_dim']

        lr = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
        den = [1]
        seq = [1,2,3,4,5]
        inte = [2,3,4,5]
        hdn = [1,3,5,7,10,20,40,80]
        for d in den:
            for s in seq:
                for i in inte:
                    for h in hdn:
                        for l in lr:
                            print(self.cnt)

                            self.min_loss = 10

                            self.LR = l
                            self.denominator = d
                            self.seq_length = s
                            self.interval = i
                            self.hidden_dim = h

                            param = {'denominator': self.denominator, 'LR': self.LR, 'seq_length': self.seq_length, 'interval': self.interval,
                                     'hidden_dim': self.hidden_dim, 'BATCH_SIZE': self.BATCH_SIZE,
                                     'iterations': self.iterations, 'data_dim': self.data_dim, 'output_dim': self.output_dim}

                            if not os.path.exists("CKPT/" + str(self.cnt) + " Save data"):  # 디렉터리가 없으면 생성한다.
                                os.makedirs("CKPT/" + str(self.cnt) + " Save data")
                            with open("CKPT/" + str(self.cnt) + " Save data/Hyper parameters.txt", 'wt') as fp:  # 생성한 디렉터리에 Hparams 정보 저장
                                json.dump(param, fp)

                            self.train()

                            self.cnt += 1


    def train(self):
        for file in self.csv_file:
            self.set_dataset(file)
            self.tf_init()
            self.training(file)

    def set_dataset(self, filename):
        json_filename = filename[20:] + " " + str(self.interval) + " " + str(self.seq_length) + " " + ".json"

        with open("DATA/seqed data/"+json_filename, 'rt') as fp:
            data = json.load(fp)
            self.cnt_num_label = data[0]
            self.cnt_label = [0 for _ in range(self.output_dim)]
            self.input =  np.array(data[1])
            self.label = np.array(data[2])
            fp.close()
        self.train_size = len(self.label)

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

    def training(self, filename):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with tf.Session() as sess:
            print("class 분포[ -, 0, + ] : ", self.cnt_num_label)  # 전체 분포
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())  # the local var is for accuracy_op
            sess.run(init_op)  # initialize var in graph

            ckpt = tf.train.get_checkpoint_state("CKPT/" + str(self.cnt)+' Save data/')
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
                #print("cnt_label : ", cnt_label)
                batch_input = batch_input[1:]
                batch_label = batch_label[1:]

                _, loss_ = sess.run([self.train_op, self.loss], {self.tf_x: batch_input, self.tf_y: batch_label})

                if self.min_loss > loss_:
                    print("------------------------")
                    print("step             : %d" % step)
                    print('* minimal loss   : %.4f' % loss_)
                    save_path = self.saver.save(sess, "CKPT/" + str(self.cnt) + " Save data/RNN-model.ckpt")
                    self.min_loss = loss_

                if step % 100 == 0:
                    print("step             : %d" % step)


        print("------------------------")
        #print(save_path)
        print("* result         : ", filename)
        print('* train loss     : %.4f' % self.min_loss)
        print("------------------------")
        # -- End of Training --
        tf.reset_default_graph()


tf_ins = Training()