"""
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
'''
# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()
'''
# Hyper Parameters=
BATCH_SIZE = 64
LR = 0.01               # learning rate

seq_length = 2
data_dim = 0
hidden_dim = 20
output_dim = 3
learning_rate = 0.01
iterations = 100

filename = "RNNsample.csv"
'''
with open(filename, 'rt') as fp: # TODO np.loadtxt로 아래에서 한 번에 할 수 있지 않을까.
    for line in fp:
        line = line[:-2].split(',')
        data_dim = hidden_dim = len(line) - 2  # 전체 호가 개수 - (time, now)
        break
'''
xy = np.loadtxt(filename, delimiter=',')#, usecols=range(data_dim + 2))  # numpy.ndarray 타입
data_dim = len(xy[0]) - 2
print(data_dim)

x = xy[..., 2:]  # 시간, 현재 가격을 제외한 호가 항목
y = xy[..., 1]  # 현재 가격*

input = np.empty((1, seq_length, data_dim), float)
label = np.empty((1, output_dim), int)

# label을, 현재가 변화량이 +일 때 +1, -일 때 -1, 0일때 0으로 초기화
for i in range(seq_length-1, len(y)-1):
    val = y[i+1] - y[i]
    if val > 0:
        lb = np.array([0, 0, 1])  # +
    elif val < 0:
        lb = np.array([1, 0, 0])  # -
    else:
        lb = np.array([0, 1, 0])  # 0
    label = np.vstack((label, lb))
label = label[1:]  # np.empty에서 생성한 label[0]을 지운다.
#print("LABEL : ", label)  # if(seq_length = 10) : len(y) = 12258 -> len(label) = 12248

print("====================================================")
# build a dataset 데이터셋 : input, label
for i in range(0, len(y) - seq_length):
    input_ = np.array([x[i:i+seq_length]])
    if i%100 == 0:
        print(i)
    input = np.vstack((input, input_))
input = input[1:]
#print("INPUT : ", input)
#print("LABEL : ", label)

#test set
train_size = int(len(label) * 0.7)
test_size = len(label) - train_size
train_input, test_input = input[0:train_size], input[train_size:len(label)]
train_label, test_label = label[0:train_size], label[train_size:len(label)]
print(test_label)
print(test_label[20])
print(test_label.shape)

# =================================================================================================================

tf_x = tf.placeholder(tf.float32, [None, seq_length, data_dim], name='tf_x')
tf_y = tf.placeholder(tf.int32, [None, output_dim], name='tf_y')

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    scope="rnn"
)
#TODO 밑의 argument에서 [:, -1, :]의 -1은 무슨 자리인지. 왜 있는지.
output = tf.layers.dense(outputs[:, -1, :], output_dim, name='dense_output')              # output based on the last output step

#outputs = tf.reshape(outputs, [-1,6])
#output = tf.contrib.layers.fully_connected(outputs, output_dim, activation_fn=None)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('Save data/')
    print(ckpt)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("## SAVED DATA LOAD ##")
        saver.restore(sess, ckpt.model_checkpoint_path)
        #saver = tf.train.import_meta_graph("Save data/RNN-model.meta")
    else:
        print("** tf.global_variables_initializer **")
        sess.run(tf.global_variables_initializer())
    #
    #saver.restore(sess, "Save data/RNN-model")
    #saver.restore(sess, tf.train.latest_checkpoint('Save data/'))
    #graph = tf.get_default_graph()

    for step in range(iterations):    # training
        batch_input = np.empty((1, seq_length, data_dim), float)
        batch_label = np.empty((1, output_dim), int)

        for i in range(BATCH_SIZE):
            idx = random.randint(0, train_size-1)
            input_ = np.array([train_input[idx]])
            label_ = np.array(train_label[idx])

            batch_input = np.vstack((batch_input, input_))
            batch_label = np.vstack((batch_label, label_))
        batch_input = batch_input[1:]
        batch_label = batch_label[1:]

        _, loss_ = sess.run([train_op, loss], {tf_x: batch_input, tf_y: batch_label})

        if step % 50 == 0:      # testing
            accuracy_ = sess.run(accuracy, {tf_x: test_input, tf_y: test_label})
            print(test_input, '->', test_label)
            print("------------------------")
            print("step             : %d" % step)
            print('* train loss     : %.4f' % loss_)
            print('* test accuracy  : %.2f' % accuracy_)

    print("------------------------")
    print("* result")
    print('* train loss     : %.4f' % loss_)
    print('* test accuracy  : %.2f' % accuracy_)
    print("------------------------")
