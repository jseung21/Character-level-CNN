import tensorflow as tf

class CharCNN_100(object):
    
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name


    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1,
                  filter_sizes=(7, 7, 3, 3, 3, 3),
                  num_filters_per_size=256,
                  num_quantized_chars=70):



        # input place holders
        self.training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        print("X : ", self.X)
        tf.summary.histogram("X", self.X)

        # ================ Layer 1 ================
        with tf.variable_scope('conv-maxpool-1'):
            conv1 = tf.layers.conv2d(inputs=self.X, filters=num_filters_per_size,
                                     kernel_size=[num_quantized_chars, filter_sizes[0]],
                                     padding="VALID", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 3],
                                            padding="VALID", strides=3)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)
            tf.summary.histogram("conv1", conv1)
            tf.summary.histogram("pool1", pool1)

        # ================ Layer 2 ================
        with tf.variable_scope('conv-maxpool-2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[1]],
                                     padding="VALID", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 3],
                                            padding="VALID", strides=3)
            # print("conv2 : ", conv2)
            # print("pool2 : ", pool2)
            tf.summary.histogram("conv2", conv2)
            tf.summary.histogram("pool2", pool2)

        # ================ Layer 3 ================
        with tf.variable_scope('conv-3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[2]],
                                     padding="VALID", activation=tf.nn.relu)
            # print("conv3 : ", conv3)
            tf.summary.histogram("conv3", conv3)

        # ================ Layer 4 ================
        with tf.variable_scope('conv-4'):
            conv4 = tf.layers.conv2d(inputs=conv3, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[3]],
                                     padding="VALID", activation=tf.nn.relu)
            # print("conv4 : ", conv4)
            tf.summary.histogram("conv4", conv4)

        # ================ Layer 5 ================
        with tf.variable_scope('conv-5'):
            conv5 = tf.layers.conv2d(inputs=conv4, filters=num_filters_per_size,
                                     kernel_size=[1, filter_sizes[4]],
                                     padding="VALID", activation=tf.nn.relu)
            # print("conv5 : ", conv5)
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
        reshape_size = 14 * 256
        if input_num_of_rows == 3:
            reshape_size *= 6
        flat = tf.reshape(pool6, [-1, reshape_size])

        with tf.variable_scope('dropout-1'):
            dropout1 = tf.layers.dropout(inputs=flat,
                                         rate=dropout_rate, training=self.training)
            # print("dropout1 : ", dropout1)
            tf.summary.histogram("dropout1", dropout1)

        with tf.variable_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=dropout1,
                                     units=1024, activation=tf.nn.relu)
            # print("fc1 : ", fc1)
            tf.summary.histogram("fc1", fc1)

        # ================ Layer 8 ================
        with tf.variable_scope('dropout-2'):
            dropout2 = tf.layers.dropout(inputs=fc1,
                                         rate=dropout_rate, training=self.training)
            # print("dropout2 : ", dropout2)
            tf.summary.histogram("dropout2", dropout2)

        with tf.variable_scope('fc-2'):
            fc2 = tf.layers.dense(inputs=dropout2,
                                  units=1024, activation=tf.nn.relu)
            # print("fc2 : ", fc2)
            tf.summary.histogram("fc2", fc2)

        # ================ Layer 9 ================
        with tf.variable_scope('fc-3'):
            fc3 = tf.layers.dense(inputs=fc2,
                                  units=1024, activation=tf.nn.relu)
            # print("fc3 : ", fc3)
            tf.summary.histogram("fc3", fc3)

        # Logits
        self.logits = tf.layers.dense(inputs=fc3, units=num_of_classes)
        tf.summary.histogram("logits", self.logits)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_110(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1,
                  filter_sizes=(3, 3, 3, 3, 3, 3),
                  num_filters_per_size=256,
                  num_quantized_chars=70):
        # input place holders
        self.training = tf.placeholder(tf.bool)
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        print("X : ", self.X)
        tf.summary.histogram("X", self.X)

        # ================ Layer 1 ================
        with tf.variable_scope('conv-maxpool-1'):
            conv1 = tf.layers.conv2d(inputs=self.X, filters=128,
                                     kernel_size=[3, filter_sizes[0]],
                                     padding="VALID", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="VALID", strides=2)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)
            tf.summary.histogram("conv1", conv1)
            tf.summary.histogram("pool1", pool1)

        # ================ Layer 2 ================
        with tf.variable_scope('conv-maxpool-2'):
            conv2 = tf.layers.conv2d(inputs=pool1, filters=128,
                                     kernel_size=[3, filter_sizes[1]],
                                     padding="VALID", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="VALID", strides=2)
            print("conv2 : ", conv2)
            print("pool2 : ", pool2)
            tf.summary.histogram("conv2", conv2)
            tf.summary.histogram("pool2", pool2)

        # ================ Layer 3 ================
        with tf.variable_scope('conv-3'):
            conv3 = tf.layers.conv2d(inputs=pool2, filters=256,
                                     kernel_size=[3, filter_sizes[2]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv3 : ", conv3)
            tf.summary.histogram("conv3", conv3)

        # ================ Layer 4 ================
        with tf.variable_scope('conv-4'):
            conv4 = tf.layers.conv2d(inputs=conv3, filters=128,
                                     kernel_size=[3, filter_sizes[3]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv4 : ", conv4)
            tf.summary.histogram("conv4", conv4)

        # ================ Layer 5 ================
        with tf.variable_scope('conv-5'):
            conv5 = tf.layers.conv2d(inputs=conv4, filters=64,
                                     kernel_size=[3, filter_sizes[4]],
                                     padding="VALID", activation=tf.nn.relu)
            print("conv5 : ", conv5)
            tf.summary.histogram("conv5", conv5)

        # ================ Layer 6 ================
        with tf.variable_scope('conv-maxpool-6'):
            conv6 = tf.layers.conv2d(inputs=conv5, filters=32,
                                     kernel_size=[3, filter_sizes[5]],
                                     padding="VALID", activation=tf.nn.relu)
            pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2],
                                            padding="VALID", strides=2)
            print("conv6 : ", conv6)
            print("pool6 : ", pool6)
            tf.summary.histogram("conv6", conv6)
            tf.summary.histogram("pool6", pool6)

        # ================ Layer 7 ================
        conv1x1 = tf.layers.conv2d(inputs=pool6, filters=16,
                                 kernel_size=[1, 1],
                                 padding="VALID", activation=tf.nn.relu)

        print("conv1x1 : ", conv1x1)

        reshape_size = 4 * 57* 16
        if input_num_of_rows == 3:
            reshape_size *= 6
        flat = tf.reshape(conv1x1, [-1, reshape_size])

        with tf.variable_scope('dropout-1'):
            dropout1 = tf.layers.dropout(inputs=flat,
                                         rate=dropout_rate, training=self.training)
            # print("dropout1 : ", dropout1)
            tf.summary.histogram("dropout1", dropout1)

        with tf.variable_scope('fc-1'):
            fc1 = tf.layers.dense(inputs=dropout1,
                                  units=1024, activation=tf.nn.relu)
            # print("fc1 : ", fc1)
            tf.summary.histogram("fc1", fc1)

        # ================ Layer 8 ================
        with tf.variable_scope('dropout-2'):
            dropout2 = tf.layers.dropout(inputs=fc1,
                                         rate=dropout_rate, training=self.training)
            # print("dropout2 : ", dropout2)
            tf.summary.histogram("dropout2", dropout2)

        with tf.variable_scope('fc-2'):
            fc2 = tf.layers.dense(inputs=dropout2,
                                  units=1024, activation=tf.nn.relu)
            # print("fc2 : ", fc2)
            tf.summary.histogram("fc2", fc2)


        # Logits
        self.logits = tf.layers.dense(inputs=fc2, units=num_of_classes)
        tf.summary.histogram("logits", self.logits)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_200(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        
    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):


        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=dropout_rate, training=self.training)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)
            # print("dropout1 : ", dropout1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=dropout_rate, training=self.training)
            # print("conv2 : ", conv2)
            # print("pool2 : ",  pool2)
            # print("dropout2 : ", dropout2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=dropout_rate, training=self.training)
            # print("conv3 : ", conv3)
            # print("pool3 : ",  pool3)
            print("dropout3 : ", dropout3)

        with tf.variable_scope("FC"):
            reshape_size = 9 * 63 * 128
            if input_num_of_rows == 3:
                reshape_size *= 3
            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, reshape_size])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=dropout_rate, training=self.training)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dropout4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)


        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary],
                             feed_dict={self.X: x_data, self.Y: y_data,
                                        self.training: training})

class CharCNN_201(object):

    def __init__(self, sess, name, first_decay_steps):
        self.sess = sess
        self.name = name
        self.first_decay_steps = first_decay_steps

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=dropout_rate, training=self.training)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)
            # print("dropout1 : ", dropout1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=dropout_rate, training=self.training)
            # print("conv2 : ", conv2)
            # print("pool2 : ",  pool2)
            # print("dropout2 : ", dropout2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=dropout_rate, training=self.training)
            # print("conv3 : ", conv3)
            # print("pool3 : ",  pool3)
            print("dropout3 : ", dropout3)

        with tf.variable_scope("FC"):
            reshape_size = 9 * 63 * 128
            if input_num_of_rows == 3:
                reshape_size *= 3
            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, reshape_size])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=dropout_rate, training=self.training)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dropout4, units=num_of_classes)

        # for return
        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        # useing cosine_decay_restarts
        self.global_step = tf.placeholder(tf.int32)
        lr_decayed = tf.train.cosine_decay_restarts(learning_rate, self.global_step,
                                                    self.first_decay_steps)

        # tf.summary.scalar("learning_rate", lr_decayed)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=lr_decayed).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, global_step, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary],
                             feed_dict={self.X: x_data, self.Y: y_data,
                                        self.training: training,
                                        self.global_step: global_step})

class CharCNN_210(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            net1 = tf.nn.relu(bn1)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=net1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=dropout_rate, training=self.training)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)
            # print("dropout1 : ", dropout1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            net2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling2d(inputs=net2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=dropout_rate, training=self.training)
            # print("conv2 : ", conv2)
            # print("pool2 : ",  pool2)
            # print("dropout2 : ", dropout2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="SAME")

            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            net3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling2d(inputs=net3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=dropout_rate, training=self.training)
            print("conv3 : ", conv3)
            print("pool3 : ",  pool3)
            print("dropout3 : ", dropout3)

        with tf.variable_scope("FC"):
            reshape_size = 9 * 63 * 128
            if input_num_of_rows == 3:
                reshape_size *= 3
            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, reshape_size])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=dropout_rate, training=self.training)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dropout4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_211(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            net1 = tf.nn.relu(bn1)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=net1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            net2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling2d(inputs=net2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            # print("conv2 : ", conv2)
            # print("pool2 : ",  pool2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     padding="SAME")

            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            net3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling2d(inputs=net3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv3 : ", conv3)
            print("pool3 : ",  pool3)

        with tf.variable_scope("FC"):
            reshape_size = 9 * 63 * 128
            if input_num_of_rows == 3:
                reshape_size *= 3
            # Dense Layer with Relu
            flat = tf.reshape(pool3, [-1, reshape_size])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dense4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_212(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            net1 = tf.nn.relu(bn1)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=net1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            # print("conv1 : ", conv1)
            # print("pool1 : ",  pool1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            net2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling2d(inputs=net2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            # print("conv2 : ", conv2)
            # print("pool2 : ",  pool2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     padding="SAME")

            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            net3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling2d(inputs=net3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv3 : ", conv3)
            print("pool3 : ",  pool3)

        with tf.variable_scope("FC"):
            reshape_size = 9 * 63 * 128
            if input_num_of_rows == 3:
                reshape_size *= 3
            # Dense Layer with Relu
            flat = tf.reshape(pool3, [-1, reshape_size])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dense4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))


        # update_ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.cost)


        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_220(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)
        print("X : ", self.X)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3, 3],
                                     padding="same")

            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            net1 = tf.nn.relu(bn1)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=net1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3],
                                     padding="SAME")

            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            net2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling2d(inputs=net2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv2 : ", conv2)
            print("pool2 : ",  pool2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            net3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling2d(inputs=net3, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            print("conv3 : ", conv3)
            print("pool3 : ", pool3)

        with tf.variable_scope("Layer4"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn4 = tf.layers.batch_normalization(conv4, training=self.training)
            net4 = tf.nn.relu(bn4)

            pool4 = tf.layers.max_pooling2d(inputs=net4, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv3 : ", conv4)
            print("pool3 : ",  pool4)

            # conv1x1 = tf.layers.conv2d(inputs=conv4, filters=16,
            #                            kernel_size=[1, 1],
            #                            padding="SAME", activation=tf.nn.relu)
            #
            # print("conv1x1 : ",  conv1x1)

        with tf.variable_scope("FC"):
            # Dense Layer with Relu
            flat = tf.reshape(pool4, [-1, 14 * 32 * 64])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dense4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class CharCNN_221(object):

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def build_net(self,
                  learning_rate=0.000001,
                  dropout_rate=0.5,
                  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n",
                  char_max_length=500,
                  num_of_classes=3,  # 0 : nothing, 1 : data (table, fixed, delimiter), 2 : key-value
                  input_num_of_rows=1):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, len(alphabet) * input_num_of_rows, char_max_length, 1])
        self.Y = tf.placeholder(tf.float32, [None, num_of_classes])
        self.training = tf.placeholder(tf.bool)
        print("X : ", self.X)

        with tf.variable_scope('Layer1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3, 3],
                                     padding="same")

            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            net1 = tf.nn.relu(bn1)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=net1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv1 : ", conv1)
            print("pool1 : ",  pool1)

        with tf.variable_scope("Layer2"):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3],
                                     padding="SAME")

            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            net2 = tf.nn.relu(bn2)

            pool2 = tf.layers.max_pooling2d(inputs=net2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv2 : ", conv2)
            print("pool2 : ",  pool2)

        with tf.variable_scope("Layer3"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            net3 = tf.nn.relu(bn3)

            pool3 = tf.layers.max_pooling2d(inputs=net3, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            print("conv3 : ", conv3)
            print("pool3 : ", pool3)

        with tf.variable_scope("Layer4"):
            # Convolutional Layer #3 and Pooling Layer #3
            conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[3, 3],
                                     padding="SAME")

            bn4 = tf.layers.batch_normalization(conv4, training=self.training)
            net4 = tf.nn.relu(bn4)

            pool4 = tf.layers.max_pooling2d(inputs=net4, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            print("conv3 : ", conv4)
            print("pool3 : ",  pool4)

            # conv1x1 = tf.layers.conv2d(inputs=conv4, filters=16,
            #                            kernel_size=[1, 1],
            #                            padding="SAME", activation=tf.nn.relu)
            #
            # print("conv1x1 : ",  conv1x1)

        with tf.variable_scope("FC"):
            # Dense Layer with Relu
            flat = tf.reshape(pool4, [-1, 14 * 32 * 64])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=1024, activation=tf.nn.relu)

        # Logits (no activation) Layer
        self.logits = tf.layers.dense(inputs=dense4, units=num_of_classes)

        values = tf.global_variables()

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))


        # update_ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.cost)

        self.logits_ = tf.argmax(self.logits, 1)
        self.Y_ = tf.argmax(self.Y, 1)
        correct_prediction = tf.equal(self.logits_, self.Y_)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()

        return values

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run([self.accuracy, self.summary, self.cost, self.logits_],
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})


