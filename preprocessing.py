import numpy as np


class Data(object):

    # def __init__(self, sess, name):
    #     self.sess = sess
    #     self.name = name

    def set_params(self, input_num_of_rows, alphabet, char_max_length):
        self.input_num_of_rows = input_num_of_rows
        self.alphabet = alphabet
        self.char_max_length = char_max_length

    def load_data(self, logger, flag):
        # TODO Add the new line character later for the yelp'cause it's a multi-line review
        examples, labels = Data.load_log(self, logger, flag)
        if self.input_num_of_rows == 3:
            examples_ = Data.modify_x(self, examples)
        elif self.input_num_of_rows == 1:
            examples_ = examples
        else:
            print('Wrong valude!')
            exit()
        x = np.array(examples_, dtype=np.int8)
        y = np.array(labels, dtype=np.int8)
        logger.info("x_char_seq_ind=" + str(x.shape))
        logger.info("y shape=" + str(y.shape))
        return [x, y]

    def modify_x(self, examples):
        x = []
        empty_row = Data.string_to_int8_conversion(Data.pad_sentence(self, list(" ")), self.alphabet)
        for i in range(len(examples)):
            if i == 0:
                x.append(np.append(np.append(empty_row,examples[i]),examples[i+1]))
            elif i == len(examples)-1:
                x.append(np.append(np.append(examples[i-1], examples[i]), empty_row))
            else:
                x.append(np.append(np.append(examples[i-1], examples[i]), examples[i + 1]))
        return x

    def load_log(self, logger, flag):
        contents = []
        labels = []

        path = './data/dataset.txt';
        if flag == 'val':
            path = './data/dataset_val.txt';

        with open(path) as f:
            i = 0
            for line in f:
                labels.append(line[0])
                text_end_extracted = Data.extract_end(self, list(" "+line[1:].lower()))
                padded = Data.pad_sentence(self, text_end_extracted)
                text_int8_repr = Data.string_to_int8_conversion(padded, self.alphabet)
                contents.append(text_int8_repr)
                i += 1
                # if i % 100 == 0:
                #     logger.info("Non-neutral instances processed: " + str(i))

        return contents, labels


    def extract_end(self, char_seq):
        if len(char_seq) > self.char_max_length:
            char_seq = char_seq[-self.char_max_length:]
        return char_seq


    def pad_sentence(self, char_seq, padding_char=" "):
        char_seq_length = self.char_max_length
        num_padding = char_seq_length - len(char_seq)
        new_char_seq = char_seq + [padding_char] * num_padding
        return new_char_seq


    def string_to_int8_conversion(char_seq, alphabet):
        x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
        return x

    def get_one_hot(batch_xs,input_num_of_rows,alphabet,char_max_length):

        if input_num_of_rows == 1:
            batch_xs_one_hot = np.zeros(shape=[len(batch_xs), len(alphabet), char_max_length, 1])
            for example_i, char_seq_indices in enumerate(batch_xs):
                for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                    if char_seq_char_ind != -1:
                        batch_xs_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
        elif input_num_of_rows == 3:
            batch_xs_one_hot = np.zeros(shape=[len(batch_xs), len(alphabet) * 3, char_max_length, 1])
            for example_i, char_seq_indices in enumerate(batch_xs):
                # char_pos_in_seq : 0~1499, char_seq_char_ind : 0~70
                for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
                    if char_seq_char_ind != -1:
                        batch_xs_one_hot[example_i][char_seq_char_ind][char_pos_in_seq % char_max_length][0] = 1
        return batch_xs_one_hot