import tensorflow as tf
import numpy as np
import os
import time
import datetime

tf.set_random_seed(777)

### hyper parameters
LEARNING_RATE = 0.000005
EPOCHS = 10000
BATCH_SIZE = 500
DROPOUT_RATE = 0.5

### parameters
CHAR_MAX_LENGTH = 500
NUM_OF_CLASSES = 3
# 0 : nothing
# 1 : data (table, fixed, delimiter)
# 2 : key-value
INPUT_NUM_OF_ROWS = 3
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"




class Data:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name


    def load_data(self, flag):
        # TODO Add the new line character later for the yelp'cause it's a multi-line review
        examples, labels = Data.load_log(self, flag)
        if INPUT_NUM_OF_ROWS != 1:
            examples_ = Data.modify_x(examples)
        else:
            examples_ = examples
        x = np.array(examples_, dtype=np.int8)
        y = np.array(labels, dtype=np.int8)
        print("x_char_seq_ind=" + str(x.shape))
        print("y shape=" + str(y.shape))
        return [x, y]

    def modify_x(self):
        x = []
        empty_row = Data.string_to_int8_conversion(Data.pad_sentence(list(" ")), ALPHABET)
        for i in range(len(self)):
            if i == 0:
                x.append(np.append(np.append(empty_row,self[i]),self[i+1]))
            elif i == len(self)-1:
                x.append(np.append(np.append(self[i-1], self[i]), empty_row))
            else:
                x.append(np.append(np.append(self[i-1], self[i]), self[i + 1]))
        return x

    def load_log(self, flag):
        contents = []
        labels = []

        path = './data/dataset.txt';
        if flag == 'val':
            path = './data/dataset_val.txt';

        with open(path) as f:
            i = 0
            for line in f:
                labels.append(line[0])
                text_end_extracted = Data.extract_end(list(" "+line[1:].lower()))
                padded = Data.pad_sentence(text_end_extracted)
                text_int8_repr = Data.string_to_int8_conversion(padded, ALPHABET)
                contents.append(text_int8_repr)
                i += 1
                if i % 100 == 0:
                    print("Non-neutral instances processed: " + str(i))

        return contents, labels


    def extract_end(char_seq):
        if len(char_seq) > CHAR_MAX_LENGTH:
            char_seq = char_seq[-CHAR_MAX_LENGTH:]
        return char_seq


    def pad_sentence(char_seq, padding_char=" "):
        char_seq_length = CHAR_MAX_LENGTH
        num_padding = char_seq_length - len(char_seq)
        new_char_seq = char_seq + [padding_char] * num_padding
        return new_char_seq


    def string_to_int8_conversion(char_seq, ALPHABET):
        x = np.array([ALPHABET.find(char) for char in char_seq], dtype=np.int8)
        return x



class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)


            # input place holders
            self.X = tf.placeholder(tf.float32, [None, CHAR_MAX_LENGTH * INPUT_NUM_OF_ROWS])

            # Input Layer [INPUT_NUM_OF_ROWS x CHAR_MAX_LENGTH]
            X_i = tf.reshape(self.X, [-1, INPUT_NUM_OF_ROWS, CHAR_MAX_LENGTH, 1])
            self.Y = tf.placeholder(tf.float32, [None, NUM_OF_CLASSES])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_i, filters=32, kernel_size=[2, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)
            print("dropout1 : ", dropout1)


            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[2, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv2 : ", conv2)
            print("pool2 : ",  pool2)
            print("dropout2 : ", dropout2)


            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[1, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv3 : ", conv3)
            print("pool3 : ",  pool3)
            print("dropout3 : ", dropout3)


            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 1 * 63])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=DROPOUT_RATE, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=NUM_OF_CLASSES)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()


    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})


class Model_2:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)


            # input place holders
            self.X = tf.placeholder(tf.float32, [None, CHAR_MAX_LENGTH * INPUT_NUM_OF_ROWS])

            # Input Layer [INPUT_NUM_OF_ROWS x CHAR_MAX_LENGTH]
            X_i = tf.reshape(self.X, [-1, INPUT_NUM_OF_ROWS, CHAR_MAX_LENGTH, 1])
            self.Y = tf.placeholder(tf.float32, [None, NUM_OF_CLASSES])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_i, filters=64, kernel_size=[2, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)
            print("dropout1 : ", dropout1)


            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=128, kernel_size=[2, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv2 : ", conv2)
            print("pool2 : ",  pool2)
            print("dropout2 : ", dropout2)


            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=256, kernel_size=[1, 3],
                                     padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv3 : ", conv3)
            print("pool3 : ",  pool3)
            print("dropout3 : ", dropout3)


            # Convolutional Layer #3 and Pooling Layer #3
            conv4 = tf.layers.conv2d(inputs=dropout2, filters=512, kernel_size=[1, 3],
                                     padding="same", activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout4 = tf.layers.dropout(inputs=pool4,
                                         rate=DROPOUT_RATE, training=self.training)
            print("conv4 : ", conv4)
            print("pool4 : ",  pool4)
            print("dropout4 : ", dropout4)

            # Dense Layer with Relu
            flat = tf.reshape(dropout4, [-1, 512 * 1 * 63])
            dense5 = tf.layers.dense(inputs=flat,
                                     units=4096, activation=tf.nn.relu)
            dropout5 = tf.layers.dropout(inputs=dense5,
                                         rate=DROPOUT_RATE, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout5, units=NUM_OF_CLASSES)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()


    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})




def check_time(time):
    second = str(int(time % 60))
    time /= 60
    min = str(int(time % 60))
    time /= 60
    hour = str(int(time))



def main():
    ### initialize
    sess = tf.Session()
    # m1 = Model(sess, "m1")
    m1 = Model_2(sess, "m1")
    dt = Data(sess, "dt")

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    saver = tf.train.Saver()

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    train_out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "train"))
    val_out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "val"))
    print("Writing to {}\n".format(out_dir))

    ### Load data
    print("Loading data...")
    x, _y = dt.load_data("train")

    # shuffle_indices = np.random.permutation(np.arange(len(x)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = _y[shuffle_indices]

    # targets = np.array(_y).reshape(-1)
    y = np.eye(NUM_OF_CLASSES)[_y]

    # n_dev_samples = int(len(x) * 0.7)
    # x_train, x_val = x[:n_dev_samples], x[n_dev_samples:]
    # y_train, y_val = y[:n_dev_samples], y[n_dev_samples:]

    x_val, y_test = dt.load_data("val")
    y_val = np.eye(NUM_OF_CLASSES)[y_test]

    x_train = x
    y_train = y

    print("Train/Validate split: {:d}/{:d}".format(len(y_train), len(y_val)))

    # Create summary writer
    # writer = tf.summary.FileWriter(out_dir)
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    # writer.add_graph(sess.graph)
    global_step = 0

    # train my model
    for epoch in range(EPOCHS):
        avg_cost = 0
        total_batch = int(len(x_train) / BATCH_SIZE)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + BATCH_SIZE]
            batch_ys = y_train[batch_cnt: batch_cnt + BATCH_SIZE]
            batch_cnt += BATCH_SIZE - 1

            c, _, train_loss_summary = m1.train(batch_xs, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)

            global_step += 1

            # print('     Batch_num:', '%04d' % (i + 1), 'loss =', '{:.9f}'.format(avg_cost))

        # validation
        accuracy, val_loss_summary, val_loss = m1.get_accuracy(x_val, y_val)
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        print('Epoch:', '%04d' % (epoch + 1)
              , '  train_loss =', '{:.9f}'.format(avg_cost)
              , '  val_loss =', '{:.9f}'.format(val_loss)
              , '  accuracy =', '{:.9f}'.format(accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))

    print('Learning Finished!')

    # Test model and check accuracy
    # accuracy, _ = m1.get_accuracy(x_val, y_val)
    print('Accuracy:', accuracy)

    save_path = saver.save(sess, out_dir+"/model.ckpt")


start_time = time.time()

main()

end_time = time.time()
h,m,s = check_time(int(end_time-start_time))
print('Program end [ Total time : '+h+" Hour "+m+" Minute "+s+" Second ]")




