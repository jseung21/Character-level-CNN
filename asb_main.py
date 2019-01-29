import tensorflow as tf
import numpy as np
import argparse
import preprocessing
import os
import time
import model
import logging

tf.set_random_seed(777)

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-m', '--nn', help='Neural Network', default='cnn', type=str,required=True)
    args = config.parse_args()
    arguments = vars(args)
    return arguments

def check_time(time):
    second = str(int(time % 60))
    time /= 60
    min = str(int(time % 60))
    time /= 60
    hour = str(int(time))
    return hour,min,second

def set_log(log_level):
    logger = logging.getLogger('5takulogger')
    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    fileHandler = logging.FileHandler('./log/'+str(int(time.time()))+'_process.log')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    level = logging.getLevelName(log_level)
    logger.setLevel(level)

    return logger

def write_params(logger, dir_name, dir, params, values):
    p1 = '### Parameters ----------------------------------'
    logger.info(p1)
    file = open(dir + '/' + dir_name + '_params.log', 'w')
    file.write(p1+'\n')
    for key in params.keys():
        value = params[key]
        _str = "\t{} : {}".format(key, value)
        logger.info(_str)
        file.write(_str + '\n')

    p2 = "### Values of Graph  ----------------------------------"
    logger.info(p2)
    file.write(p2+'\n')
    for val in values:
        _str = '\t'+str(val)
        logger.info(_str)
        file.write(_str + '\n')

    p3 = '--------------------------------------------------------------------'
    logger.info(p3)
    file.write(p3 + '\n')
    file.close()

def make_dir(logger, model):
    timestamp = model + "_" +  str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    train_out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "train"))
    val_out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "val"))
    logger.info("Writing to {}".format(out_dir))
    return timestamp, out_dir, train_out_dir, val_out_dir

def cal_each_class(logger, batch_ys, y_predict):
    # print('type(batch_ys):{}'.format(type(batch_ys)))
    # print(batch_ys)
    # print(np.argmax(batch_ys, 1))
    # print(y_predict)

    label = np.argmax(batch_ys, 1)
    total_0 = 0
    positive_0 = 0
    negative_0 = 0
    total_1 = 0
    positive_1 = 0
    negative_1 = 0
    total_2 = 0
    positive_2 = 0
    negative_2 = 0
    for i in range(len(label)):
        if label[i] == 0:
            total_0 += 1
            if y_predict[i] == 0:
                positive_0 += 1
            elif y_predict[i] == 1:
                negative_1 += 1
            elif y_predict[i] == 2:
                negative_2 += 1
        if label[i] == 1:
            total_1 += 1
            if y_predict[i] == 0:
                negative_0 += 1
            elif y_predict[i] == 1:
                positive_1 += 1
            elif y_predict[i] == 2:
                negative_2 += 1
        if label[i] == 2:
            total_2 += 1
            if y_predict[i] == 0:
                negative_0 += 1
            elif y_predict[i] == 1:
                negative_1 += 1
            elif y_predict[i] == 2:
                positive_2 += 1

    precision_0 = positive_0 / (positive_0 + negative_0)
    precision_1 = positive_1 / (positive_1 + negative_1)
    precision_2 = positive_2 / (positive_2 + negative_2)

    recall_0 = positive_0/total_0
    recall_1 = positive_1/total_1
    recall_2 = positive_2/total_2

    fScore_0 = (precision_0 * recall_0) / (precision_0 + recall_0)
    fScore_1 = (precision_1 * recall_1) / (precision_1 + recall_1)
    fScore_2 = (precision_2 * recall_2) / (precision_0 + recall_2)

    logger.info('precision_0:[{:.9f}], precision_1:[{:.9f}], precision_2:[{:.9f}]'
                .format(precision_0, precision_1, precision_2))
    logger.info('recall_0:[{:.9f}], recall_1:[{:.9f}], recall:[{:.9f}]'
                .format(recall_0, recall_1, recall_2))
    logger.info('fScore_0:[{:.9f}], fScore_1:[{:.9f}], fScore_2:[{:.9f}]'
                .format(fScore_0, fScore_1, fScore_2))
    logger.info('f1Score_0:[{:.9f}], f1Score_1:[{:.9f}], f1Score_2:[{:.9f}]'
                .format(fScore_0*2, fScore_1*2, fScore_2*2))



def train_model_100(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows,
                    filter_sizes,
                    num_filters_per_size,
                    num_quantized_chars):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows,
              'filter_sizes': filter_sizes,
              'num_filters_per_size': num_filters_per_size,
              'num_quantized_chars': num_quantized_chars}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_100(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows,
                              filter_sizes,
                              num_filters_per_size,
                              num_quantized_chars)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))

    # save_path = saver.save(sess, out_dir+"/model.ckpt")


    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_110(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows,
                    filter_sizes,
                    num_filters_per_size,
                    num_quantized_chars):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows,
              'filter_sizes': filter_sizes,
              'num_filters_per_size': num_filters_per_size,
              'num_quantized_chars': num_quantized_chars}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_110(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows,
                              filter_sizes,
                              num_filters_per_size,
                              num_quantized_chars)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))

    # save_path = saver.save(sess, out_dir+"/model.ckpt")


    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_200(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_200(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1

        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val
            cal_each_class(logger, batch_ys, y_predict)

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_201(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows,
                    first_decay_steps):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows,
              'first_decay_steps': first_decay_steps}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_201(sess, model_name, first_decay_steps)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys, global_step)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1

        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

            cal_each_class(logger, batch_ys, y_predict)

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_210(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_210(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))
        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_211(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_211(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))
        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_212(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_212(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))
        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_220(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_220(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')

def train_model_221(model_name, logger,
                    x_train, y_train,
                    x_val, y_val,
                    epochs, batch_size,
                    learning_rate,
                    dropout_rate,
                    alphabet,
                    char_max_length,
                    num_of_classes,
                    input_num_of_rows):

    params = {'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'dropout_rate': dropout_rate,
              'alphabet': alphabet,
              'char_max_length': char_max_length,
              'num_of_classes': num_of_classes,
              'input_num_of_rows': input_num_of_rows}

    logger.info('### ' + model_name + ' Learning Start!')
    sess = tf.Session()

    model_ = model.CharCNN_221(sess, model_name)
    values = model_.build_net(learning_rate,
                              dropout_rate,
                              alphabet,
                              char_max_length,
                              num_of_classes,
                              input_num_of_rows)

    # session run
    sess.run(tf.global_variables_initializer())

    ### tensor model save ###
    # saver = tf.train.Saver()

    # train model
    # Output directory for models and summaries
    dir_name, out_dir, train_out_dir, val_out_dir = make_dir(logger, model_name)

    # Create summary writer
    train_writer, val_writer = tf.summary.FileWriter(train_out_dir), tf.summary.FileWriter(val_out_dir)
    train_writer.add_graph(sess.graph)

    # write_params
    write_params(logger, dir_name, out_dir, params, values)

    global_step = 0
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        batch_cnt = 0
        for i in range(total_batch):
            batch_xs = x_train[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_train[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            c, _, train_loss_summary = model_.train(batch_xs_one_hot, batch_ys)
            avg_cost += c / total_batch

            # tensorboard
            train_writer.add_summary(train_loss_summary, global_step=global_step)
            global_step += 1


        # validation
        avg_val_loss = 0
        avg_accuracy = 0
        total_batch_val = int(len(x_val) / batch_size)
        batch_cnt = 0
        for i in range(total_batch_val):
            batch_xs = x_val[batch_cnt: batch_cnt + batch_size]
            batch_ys = y_val[batch_cnt: batch_cnt + batch_size]
            batch_cnt += batch_size - 1

            batch_xs_one_hot = preprocessing.Data.get_one_hot(batch_xs,input_num_of_rows,
                                                              alphabet,char_max_length)

            accuracy, val_loss_summary, val_loss, y_predict = model_.get_accuracy(batch_xs_one_hot, batch_ys)
            avg_val_loss += val_loss / total_batch_val
            avg_accuracy += accuracy / total_batch_val

        # tensorboard
        val_writer.add_summary(val_loss_summary, global_step=global_step)

        logger.info("Epoch:{:4d}".format(epoch+1)
                    + ",  train_loss={:.9f}".format(avg_cost)
                    + ",  val_loss={:.9f}".format(avg_val_loss)
                    + ",  accuracy={:.9f}".format(avg_accuracy))

        # save_path = saver.save(sess, out_dir + "/model.ckpt" + str(epoch))


    # save_path = saver.save(sess, out_dir+"/model.ckpt")

    # close
    tf.reset_default_graph()
    sess.close()

    logger.info('### ' + model_name + ' Learning Finished!')


def main(model_run):

    ### initialize ###
    # sess = tf.Session()   Note:It need each model

    # args = user_input()
    # nn = args['m']

    # logger setting, Logger Level [DEBUG, INFO(Default), WARNING, ERROR, CRITICAL]
    logger = set_log('DEBUG')


    ### parameters ###
    # hyper params
    learning_rate = 0.000005
    epochs = 300
    batch_size = 128
    dropout_rate = 0.5

    ### parameters
    char_max_length = 500
    num_of_classes = 3  # 0 : nothing # 1 : data (table, fixed, delimiter) # 2 : key-value
    input_num_of_rows = 1
    alphabet= "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"


    ### preprocessing ###
    # dt = preprocessing.Data(sess, "dataPreprocessing")
    dt = preprocessing.Data()
    dt.set_params(input_num_of_rows, alphabet, char_max_length)

    ### Load data ###
    logger.info('### Loading data...')

    x, _y = dt.load_data(logger, "train")
    y = np.eye(num_of_classes)[_y]

    x_val, y_test = dt.load_data(logger, "val")
    y_val = np.eye(num_of_classes)[y_test]

    x_train = x
    y_train = y

    logger.info(str("Train/Validate split: {:d}/{:d}".format(len(y_train), len(y_val))))

    ## model 100 ###
    model_name = 'model_100'
    if model_run[model_name]:
        ep = [1000, 2000]
        lr = [0.0000001, 0.00000001]
        for i in range(2):
            start_time = time.time()
            train_model_100(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=ep[i], batch_size=128,
                            learning_rate=lr[i],
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=input_num_of_rows,
                            filter_sizes=(7, 7, 3, 3, 3, 3),
                            num_filters_per_size=256,
                            num_quantized_chars=70)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ## model 110 ###
    model_name = 'model_110'
    if model_run[model_name]:
        start_time = time.time()
        train_model_110(model_name, logger,
                        x_train, y_train,
                        x_val, y_val,
                        1, 64,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                        alphabet=alphabet,
                        char_max_length=char_max_length,
                        num_of_classes=num_of_classes,
                        input_num_of_rows=1,
                        filter_sizes=(3, 3, 3, 3, 3, 3),
                        num_filters_per_size=256,
                        num_quantized_chars=70)
        end_time = time.time()
        h, m, s = check_time(int(end_time - start_time))
        logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ### model 200 ###
    model_name = 'model_200'
    if model_run[model_name]:
        start_time = time.time()
        train_model_200(model_name, logger,
                        x_train, y_train,
                        x_val, y_val,
                        epochs=300, batch_size=128,
                        learning_rate=0.0001,
                        dropout_rate=dropout_rate,
                        alphabet=alphabet,
                        char_max_length=char_max_length,
                        num_of_classes=num_of_classes,
                        input_num_of_rows=input_num_of_rows)
        end_time = time.time()
        h, m, s = check_time(int(end_time - start_time))
        logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ### model 201 ###
    model_name = 'model_201'
    if model_run[model_name]:
        ep = [500, 500]
        lr = [0.001, 0.0001]
        fds = [10000, 10000]
        for i in range (1):
            start_time = time.time()
            train_model_201(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=1000, batch_size=128,
                            learning_rate=0.001,
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=input_num_of_rows,
                            first_decay_steps=5000)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ### model 210 ###
    model_name = 'model_210'
    if model_run[model_name]:
        start_time = time.time()
        train_model_210(model_name, logger,
                        x_train, y_train,
                        x_val, y_val,
                        300, 128,
                        learning_rate=0.001,
                        dropout_rate=dropout_rate,
                        alphabet=alphabet,
                        char_max_length=char_max_length,
                        num_of_classes=num_of_classes,
                        input_num_of_rows=1)
        end_time = time.time()
        h, m, s = check_time(int(end_time - start_time))
        logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ### model 211 ###
    model_name = 'model_211'
    if model_run[model_name]:
        ep = [500, 500]
        lr = [0.05, 0.01]
        for i in range(2):
            start_time = time.time()
            train_model_211(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=ep[i], batch_size=128,
                            learning_rate=lr[i],
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=1)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ### model 212 ###
    model_name = 'model_212'
    if model_run[model_name]:
        ep = [1000]
        lr = [0.000001]
        for i in range(1):
            start_time = time.time()
            train_model_212(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=ep[i], batch_size=128,
                            learning_rate=lr[i],
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=1)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ## model 220 ###
    model_name = 'model_220'
    if model_run[model_name]:
        ep = [1000]
        lr = [0.00001]
        for i in range(2):
            start_time = time.time()
            train_model_220(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=ep[i], batch_size=128,
                            learning_rate=lr[i],
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=3)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")

    ## model 221 ###
    model_name = 'model_221'
    if model_run[model_name]:
        ep = [1000]
        lr = [0.000001]
        for i in range(2):
            start_time = time.time()
            train_model_221(model_name, logger,
                            x_train, y_train,
                            x_val, y_val,
                            epochs=ep[i], batch_size=128,
                            learning_rate=lr[i],
                            dropout_rate=dropout_rate,
                            alphabet=alphabet,
                            char_max_length=char_max_length,
                            num_of_classes=num_of_classes,
                            input_num_of_rows=3)
            end_time = time.time()
            h, m, s = check_time(int(end_time - start_time))
            logger.info(model_name + " Program end [ Total time : " + h + " Hour " + m + " Minute " + s + " Second ]")


# model_100 : c p c p c c c c p d f d f f
# model_110 : c p c p c c c c p c1 d f d f f
# model_200 : c p d c p d c p d f d
# model_201 : apply cosine_decay_restarts to model_200
# model_210 : c b p d c b p d c b p d f d (apply bn)
# model_211 : c b p c b p c b p f           (remove dropout)
# model_212 : apply tf.GraphKeys.UPDATE_OPS to model_211
# model_220 : c b c b c b c b p c1 f                (only 3 rows)
# model_221 : apply tf.GraphKeys.UPDATE_OPS to model_220

main(model_run = {'model_100': False,
                  'model_110': False,
                  'model_200': False,
                  'model_201': True,
                  'model_210': False,
                  'model_211': False,
                  'model_212': False,
                  'model_220': False,
                  'model_221': False})