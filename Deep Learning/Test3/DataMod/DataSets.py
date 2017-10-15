import numpy as np
from DataMod.User import User


# this class is responsible for splitting the data into training and validation.
class DataSets(object):
    def __init__(self,  ratio, ratio_validation, user):
        np.random.seed(0)
        self.user = user
        self.users = DataSets.__get_users(ratio)
        self.is_validation = DataSets.__get_validations(self.users, ratio_validation)
        self.training_index = 0
        self.validation_index = 0


    # returns all data from the next user. It strips out the validation records.
    def get_training(self):
        if self.training_index >= self.users.shape[0]:
            self.training_index = 0
        predictors, labels = self.user.get_data(self.users[self.training_index, 0])

        if self.is_validation[self.training_index, 0]:
            predictors = predictors[0:(predictors.shape[0]-1)]
            labels = labels[0:(labels.shape[0] - 1)]
        self.training_index += 1
        return predictors, labels

    def get_validation(self):
        if self.validation_index >= self.users.shape[0]:
            return None, None
        while True:
            if self.is_validation[self.validation_index, 0]:
                predictors, labels = self.user.get_data(self.users[self.validation_index, 0])
                self.validation_index += 1
                return predictors, labels
            else:
                self.validation_index += 1
                if self.validation_index >= self.users.shape[0]:
                    return None, None

    @staticmethod
    def __get_users(used_count):
        count_users = User.max_user
        # count_users_split = used_count
        # users = np.array(range(1, count_users +1)).reshape((count_users, 1))
        # users_cut = DataSets.__perm_list(count_users_split, users)
        users_cut = np.array(range(1, count_users + 1)).reshape((count_users, 1))
        return users_cut

    @staticmethod
    def __get_validations(users, ratio_validation):
        count_validation = np.floor(users.shape[0] * ratio_validation).astype(int)
        user_list = np.arange(0, users.shape[0], 1)
        validations = DataSets.__perm_list(count_validation, user_list).reshape(count_validation)
        is_validation = np.zeros(( users.shape[0], 1))
        is_validation[np.ix_(validations)] = 1
        return is_validation

    @staticmethod
    def __perm_list(count, list):
        seq = np.arange(0, len(list), 1)
        permutation = np.random.permutation(seq)
        permutation = permutation[0:count]
        outptut = list[np.ix_(permutation)]
        return outptut
