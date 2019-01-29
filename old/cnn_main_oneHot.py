import tensorflow as tf
import numpy as np
import os
import time
import datetime

tf.set_random_seed(777)

### hyper parameters
LEARNING_RATE = 0.000001
EPOCHS = 500
BATCH_SIZE = 128
DROPOUT_RATE = 0.5

### parameters
CHAR_MAX_LENGTH = 500
NUM_OF_CLASSES = 3      # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
INPUT_NUM_OF_ROWS = 1
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"




class Data:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name


    def load_data(self, flag):
        # TODO Add the new line character later for the yelp'cause it's a multi-line review
        examples, labels = Data.load_log(self, flag)
        if INPUT_NUM_OF_ROWS == 3:
            examples_ = Data.modify_x(examples)
        elif INPUT_NUM_OF_ROWS == 1:
            examples_ = examples
        else:
            print('Wrong valude!')
            exit()
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

    def get_one_hot(self, batch_xs):

        if INPUT_NUM_OF_ROWS == 1:
            batch_xs_one_hot = np.zeros(shape=[len(batch_xs), len(ALPHABET), CHAR_MAX_LENGTH, 1])
            for example_i, char_seq_indices in enumerate(batch_xs):
                for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                    if char_seq_char_ind != -1:
                        batch_xs_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
        elif INPUT_NUM_OF_ROWS == 3:
            batch_xs_one_hot = np.zeros(shape=[len(batch_xs), len(ALPHABET) * 3, CHAR_MAX_LENGTH, 1])
            for example_i, char_seq_indices in enumerate(batch_xs):
                # char_pos_in_seq : 0~1499, char_seq_char_ind : 0~70
                for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                    if char_seq_char_ind != -1:
                        batch_xs_one_hot[example_i][char_seq_char_ind][char_pos_in_seq % CHAR_MAX_LENGTH][0] = 1
        return batch_xs_one_hot


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        filter_sizes = (7, 7, 3, 3, 3, 3)
        num_filters_per_size = 256
        num_quantized_chars = 70


        # input place holders
        self.training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, [None, len(ALPHABET) * INPUT_NUM_OF_ROWS, CHAR_MAX_LENGTH, 1])
        self.Y = tf.placeholder(tf.float32, [None, NUM_OF_CLASSES])
        print("X : ", self.X)
        tf.summary.histogram("X", self.X)

        # ================ Layer 1 ================
        with tf.variable_scope('conv-maxpool-1'):
            conv1 = tf.layers.conv2d(inputs=self.X, filters=num_filters_per_size,
                                     kernel_size=[num_quantized_chars, filter_sizes[0]],
                                     padding="VALID", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 3],
                                            padding="VALID", strides=3)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)
            tf.summary.histogram("conv1", conv1)
            tf.summary.histogram("pool1", pool1)

        # ================ Layer 2 ================
        with tf.variable_scope('conv-maxpool-2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[1]],
                                     padding="VALID", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 3],
                                            padding="VALID", strides=3)
            print("conv2 : ", conv2)
            print("pool2 : ", pool2)
            tf.summary.histogram("conv2", conv2)
            tf.summary.histogram("pool2", pool2)

        # ================ Layer 3 ================
        with tf.variable_scope('conv-3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[2]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv3 : ", conv3)
            tf.summary.histogram("conv3", conv3)

        # ================ Layer 4 ================
        with tf.variable_scope('conv-4'):
            conv4 = tf.layers.conv2d(inputs=conv3, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[3]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv4 : ", conv4)
            tf.summary.histogram("conv4", conv4)

        # ================ Layer 5 ================
        with tf.variable_scope('conv-5'):
            conv5 = tf.layers.conv2d(inputs=conv4, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[4]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv5 : ", conv5)
            tf.summary.histogram("conv5", conv5)

        # ================ Layer 6 ================
        with tf.variable_scope('conv-maxpool-6'):
            conv6 = tf.layers.conv2d(inputs=conv5, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[5]],
                                     padding="VALID", activation=tf.nn.relu)
            pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[1, 3],
                                            padding="VALID", strides=3)
            print("conv6 : ", conv6)
            print("pool6 : ", pool6)
            tf.summary.histogram("conv6", conv6)
            tf.summary.histogram("pool6", pool6)

        # ================ Layer 7 ================
        flat = tf.reshape(pool6, [-1, 14*256])

        with tf.variable_scope('dropout-1'):
            dropout1 = tf.layers.dropout(inputs=flat,
                                         rate=DROPOUT_RATE, training=self.training)
            print("dropout1 : ", dropout1)
            tf.summary.histogram("dropout1", dropout1)

        with tf.variable_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=dropout1,
                                     units=1024, activation=tf.nn.relu)
            print("fc1 : ", fc1)
            tf.summary.histogram("fc1", fc1)

        # ================ Layer 8 ================
        with tf.variable_scope('dropout-2'):
            dropout2 = tf.layers.dropout(inputs=fc1,
                                         rate=DROPOUT_RATE, training=self.training)
            print("dropout2 : ", dropout2)
            tf.summary.histogram("dropout2", dropout2)

        with tf.variable_scope('fc-2'):
            fc2 = tf.layers.dense(inputs=dropout2,
                                  units=1024, activation=tf.nn.relu)
            print("fc2 : ", fc2)
            tf.summary.histogram("fc2", fc2)

        # ================ Layer 9 ================
        with tf.variable_scope('fc-3'):
            fc3 = tf.layers.dense(inputs=fc2,
                                  units=1024, activation=tf.nn.relu)
            print("fc3 : ", fc3)
            tf.summary.histogram("fc3", fc3)


        # Logits
        self.logits = tf.layers.dense(inputs=fc3, units=NUM_OF_CLASSES)
        tf.summary.histogram("logits", self.logits)

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
    m1 = Model(sess, "m1")
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

    ### in case shuffle
    # shuffle_indices = np.random.permutation(np.arange(len(x)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = _y[shuffle_indices]

    # targets = np.array(_y).reshape(-1)
    y = np.eye(NUM_OF_CLASSES)[_y]

    ### in case divide data
    # n_dev_samples = int(len(x) * 0.7)
    # x_train, x_val = x[:n_dev_samples], x[n_dev_samples:]
    # y_train, y_val = y[:n_dev_samples], y[n_dev_samples:]

    x_val, y_test = dt.load_data("val")
    y_val = np.eye(NUM_OF_CLASSES)[y_test]

    x_train = x
    y_train = y

    print("Train/Validate split: {:d}/{:d}".format(len(y_train), len(y_val)))

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

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

            batch_xs_one_hot = dt.get_one_hot(batch_xs)

            c, _, train_loss_summary = m1.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        x_val_one_hot = dt.get_one_hot(x_val)
        accuracy, val_loss_summary, val_loss = m1.get_accuracy(x_val_one_hot, y_val)
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




