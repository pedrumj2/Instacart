import numpy as np


# responsible for the final preparation of the data being passed to the neural net. Such as
# grouping the matrixes in the size of the rnn_length
class RnnAux(object):
    def __init__(self, dataSet, rnn_length):
        self.dataSet = dataSet
        self.rnn_length = rnn_length
        self.predictors = None
        self.labels = None


    def get_training(self):
        """
        return the next batch of training datas
        :return: 
            pred_out: an array of dimensions rnn_length * pred_size *1
            lab_out: an array of dimensions rrn_length * item_size *2. If first column is "1" it would indicate a
                purchase.
            is_new: if this is a new users data.
        """
        is_new = False
        # all the data for the current user is read in at once. On each call of this function
        # a section of the data is returned. If all the data has been read from the user then this will
        # be null indicating data from the next user should be read
        if self.predictors is None:
            is_new = True
            self.predictors, self.labels = self.dataSet.get_training()
            self.predictors =np.reshape(self.predictors, (self.predictors.shape[0], self.predictors.shape[1], 1))
        count = self.predictors.shape[0]
        extra = count % self.rnn_length
        # indicates that the input data is not a multiple of the rnn length. Therefore extra data is padded
        # with zeros
        if extra == 0:
            pred_out = self.predictors[0:self.rnn_length]
            lab_out = self.labels[0:self.rnn_length]
            # adds a 1 as the first index of the predictors to indicate that these are not padded rows
            pred_out = np.append(np.ones((pred_out.shape[0], 1, 1)), pred_out, 1)
            self.predictors = self.predictors[self.rnn_length:(self.predictors.shape[0])]
            # basically removing the intial part from self.labels that will be returned
            self.labels = self.labels[self.rnn_length:(self.labels.shape[0])]
        else:
            pred_out = self.predictors[0:extra]
            lab_out = self.labels[0:extra]
            # adds a 1 as the first index of the predictors to indicate that these are not padded rows
            pred_out = np.append(np.ones((pred_out.shape[0],1 , 1)), pred_out,1)
            pred_out = RnnAux.__pad_data_pred(pred_out, self.rnn_length - extra)
            lab_out = RnnAux.__pad_data_lab(lab_out, self.rnn_length - extra)
            self.predictors = self.predictors[extra:(self.predictors.shape[0])]
            self.labels = self.labels[extra:(self.labels.shape[0])]
        self.__check_end()
        return pred_out, lab_out, is_new

    def get_test(self):
        is_new = False
        is_end_all = False
        if self.predictors is None:
            is_new = True
            self.predictors, self.labels = self.dataSet.get_validation()
            if self.predictors is not None:
                self.predictors = np.reshape(self.predictors, (self.predictors.shape[0], self.predictors.shape[1], 1))
            else:
                return None, None, False, False, True
        count = self.predictors.shape[0]
        extra = count % self.rnn_length
        if extra == 0:
            pred_out = self.predictors[0:self.rnn_length]
            lab_out = self.labels[0:self.rnn_length]
            pred_out = np.append(np.ones((pred_out.shape[0], 1, 1)), pred_out, 1)
            self.predictors = self.predictors[self.rnn_length:(self.predictors.shape[0])]
            self.labels = self.labels[self.rnn_length:(self.labels.shape[0])]
            is_end_user = self.__check_end()
        else:
            pred_out = self.predictors[0:extra]
            lab_out = self.labels[0:extra]
            pred_out = np.append(np.ones((pred_out.shape[0],1 , 1)), pred_out,1)
            pred_out = RnnAux.__pad_data_pred(pred_out, self.rnn_length - extra)
            lab_out = RnnAux.__pad_data_lab(lab_out, self.rnn_length - extra)
            self.predictors = self.predictors[extra:(self.predictors.shape[0])]
            self.labels = self.labels[extra:(self.labels.shape[0])]
            is_end_user = self.__check_end()
        return pred_out, lab_out, is_new, is_end_user, is_end_all

    def __check_end(self):
        if self.predictors.shape[0] == 0:
            self.predictors = None
            self.labels = None
            return True
        else:
            return False

    @staticmethod
    def __pad_data_pred(data, pad_count):
        for i in range(pad_count):
            zeros = np.zeros((1, data.shape[1], 1))
            data = np.append(zeros, data, 0)
        return data

    @staticmethod
    def __pad_data_lab(data, pad_count):
        for i in range(pad_count):
            zeros = np.ones((data.shape[1], 1))*0.000001
            ones = np.ones((data.shape[1], 1))
            labels = np.concatenate((zeros, ones), 1)
            x = labels.reshape((1, data.shape[1], 2))
            data = np.append(x, data, 0)
        return data

