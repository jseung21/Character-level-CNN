import tensorflow as tf
import numpy as np
import re
import os
import time
from tensorflow.contrib import rnn

tf.set_random_seed(777)

#  hyper param
LEARNING_RATE = 0.005
EPOCHS = 50000
BATCH_SIZE = 500
MAX_LENGTH = 500


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    # Make a lstm cell with hidden_size (each unit output vector size)
    def lstm_cell(self, hidden_size):
        cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        return cell


    def build_net(self, char_set):

        sequence_length = MAX_LENGTH
        data_dim = len(char_set)
        hidden_size = len(char_set)
        output = 1
        num_classes = len(char_set)

        with tf.variable_scope('RNN'):
            self.X = tf.placeholder(tf.int32, [None, sequence_length])
            self.Y = tf.placeholder(tf.float32, [None, output])

            self.X_1 = tf.one_hot(self.X, num_classes)
            print(self.X_1)  # check out the shape


            multi_cells = rnn.MultiRNNCell([self.lstm_cell(hidden_size) for _ in range(2)], state_is_tuple=True)

            # outputs: unfolding size x hidden size, state = hidden size
            # initail_state = multi_cells.zero_state(BATCH_SIZE, tf.float32)
            outputs, _states = tf.nn.dynamic_rnn(multi_cells, self.X_1, dtype=tf.float32)

            outputs = tf.contrib.layers.fully_connected(outputs, num_classes, activation_fn=None)

            outputs = tf.reduce_mean(outputs, axis=1)

            # FC layer
            # FC = tf.reshape(outputs, [-1, num_classes])
            outputs = tf.contrib.layers.fully_connected(outputs, 1, activation_fn=None)

        # All weights are 1 (equal weights)
        self.weights = tf.ones([BATCH_SIZE, sequence_length])

        # sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
        self.loss = tf.reduce_sum(tf.square(outputs - self.Y))  # sum of the squares
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def train(self, x_data, y_data):
        return self.sess.run([self.optimizer, self.loss], feed_dict={self.X: x_data, self.Y: y_data})

    def validation(self, x_data, y_data):
        return self.sess.run([self.loss], feed_dict={self.X: x_data, self.Y: y_data})



def check_time(time):
    second = str(int(time % 60))
    time /= 60
    min = str(int(time % 60))
    time /= 60
    hour = str(int(time))
    return hour,min,second

def data_load(flag):
    sample = []
    y = []
    maxLen = 0

    path = './data/dataset.txt'
    if flag == 'val':
        path = './data/dataset_val.txt'

    with open(path, 'r') as f:
        while True:
            data = f.readline()
            if not data: break
            y.append(data[0])
            data = re.sub(r"[\n\r]", "", str(data[1:]))
            data = re.sub(r"[^a-zA-Z0-9-,;.!?:'\"/\\|_@#$%^&*~`+-=<>(){} \n\r\t]", "*", str(data))
            sample.append(data)
            maxLen = max(maxLen, len(data))
            # print(data)

    print("Max Length : ", maxLen)
    print("Total Rows : ", len(sample))

    limit = 0
    # TODO MAX_LENGTH fix
    # MAX_LENGTH = maxLen
    fixed_sample = []
    sequence_sizes = []

    if limit > 0:
        print("Max Length Limit : ", limit)
    else:
        print("No Max Length Limit.")

    for i in range(len(sample)):
        sequence_sizes.append(len(sample[i]))
        if limit > 0:
            fixed_sample.append(sample[i].ljust(MAX_LENGTH)[:limit])
        else:
            fixed_sample.append(sample[i].ljust(MAX_LENGTH))

    y = np.array(y)
    y = np.reshape(y, (len(y), 1))

    return fixed_sample, y, sequence_sizes, sample

def char2index(char_dic, data):
    x = []
    for i, data in enumerate(data):
        lower = data.lower()

        x_vec = [char_dic[c] for c in lower]
        # print(data , " => " , x)
        x.append(x_vec)
    return x

def main():


    sess = tf.Session()
    md = Model(sess, "md")
    sess.run(tf.global_variables_initializer())



    # data load
    fixed_sample, y_train, sequence_sizes, sample = data_load('tarin')
    fixed_sample_val, y_val, sequence_sizes_val, sample_val = data_load('val')


    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} \n\r\t"
    # char_set = list(set(alphabet))

    char_set = []
    for i in range(len(alphabet)):
        char_set.append(alphabet[i:i + 1])
    char_dic = {w: i for i, w in enumerate(char_set)}

    md.build_net(char_set)

    x_train = char2index(char_dic, fixed_sample)
    x_val = char2index(char_dic, fixed_sample_val)


    iter = int(len(sample) / BATCH_SIZE)

    # Training step
    for i in range(EPOCHS):

        avg_loss = 0

        for b in range(int(iter)):

            bx = x_train[int(b * BATCH_SIZE): int((b + 1) * BATCH_SIZE)]
            by = y_train[int(b * BATCH_SIZE): int((b + 1) * BATCH_SIZE)]

            _, step_loss = md.train(bx, by)

            avg_loss += step_loss / iter

        loss_val = md.validation(x_val, y_val)

        print("[Epoch: {}]  train_loss: [{}] ".format(i, avg_loss),  "  val_loss: {} ".format(loss_val))





        # start = int(iter * BATCH_SIZE)
        # end = int(len(sample))
        #
        # if (start != end):
        #     bx = x[int(iter * BATCH_SIZE): int(len(sample))]
        #     by = y[int(iter * BATCH_SIZE): int(len(sample))]
        #
        #     _, step_loss = sess.run([train, loss], feed_dict={
        #         X: bx, Y: by})
        #
        #     if i % 100 == 0:
        #         print("[step: {}] loss: {} ".format(i, step_loss))

    print("done!")



















start_time = time.time()

main()

end_time = time.time()
h,m,s = check_time(int(end_time-start_time))
print('Program end [ Total time : '+h+" Hour "+m+" Minute "+s+" Second ]")
