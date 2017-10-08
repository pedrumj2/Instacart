import pymysql
import numpy as np


# This class gets data for a single user
class User(object):
    max_user = 206209
    pred_size = 32
    label_size = 49688

    def __init__(self,  password, ip, database, user):
        self.cnx = pymysql.connect(user=user, password=password,
                                           host=ip,
                                           database=database)
        self.cursor = self.cnx.cursor()
        self.users = range(1, self.max_user+1)

    # Returns all orders made by user on all purchases. It groups the purchases done in a single purchase into
    # a single vector. The output will be in the form (purchase_count * pred_size * 1)
    def get_data(self, user_id):
        query = "SELECT * FROM insta.combined where user_id = " + str(user_id) + " order by order_number asc, " \
                                                                            "add_to_cart_order asc "
        self.cursor.execute(query)
        predictors = np.zeros((0, 32+self.max_user))
        labels = np.zeros((0, 49688, 2))

        order_number = -1
        label = User.__new_label()
        for (x) in self.cursor:
            if order_number != x[4]:
                if order_number != -1:
                    labels = np.append(labels, label.reshape((1,  User.label_size, 2)), 0)
                order_number = x[4]
                order = User.__get_single(x)
                user_vec = User.__long_2_vec(user_id, self.max_user)
                order = np.append(order, user_vec, 1)
                predictors = np.append(predictors, order, 0)
                label = User.__new_label()
            label = User.__add_label(x, label)
        labels = np.append(labels, label.reshape((1, User.label_size, 2)), 0)

        return predictors, labels

    @staticmethod
    def __new_label():
        label_ones = np.ones((User.label_size, 1))
        label_zeros = np.ones((User.label_size, 1))*0.00001
        output = np.stack((label_zeros, label_ones), 1).reshape((User.label_size, 2))
        return output

    @staticmethod
    def __add_label(data, label):
        label[data[0]-1, 0] = 1
        label[data[0]-1, 1] = 0.00001
        return label


    @staticmethod
    def __get_single(data):
        output = np.zeros((1, 32))
        output[0, 0:7] = User.__long_2_vec(data[5]+1, 7)
        output[0, 8:32] = User.__long_2_vec(data[6] + 1, 24)
        output[0, 7] = ((data[7]-10.6659)/82.46)
        return output

    @staticmethod
    def __long_2_vec(input, max):
        temp_arr = [0.000001] * (input-1)
        temp_arr += [1]
        temp_arr += [0.000001] * (max - input)
        output = np.array(temp_arr).reshape(1, max)
        return output
