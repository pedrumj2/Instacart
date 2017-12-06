import pymysql
import numpy as np


# This class gets data for a single user
class User(object):
    pred_size = 32
    max_user = -1
    label_size = -1
    ratio_purchase = -1

    def __init__(self,  password, ip, database, user):
        self.cnx = pymysql.connect(user=user, password=password,
                                           host=ip,
                                           database=database)
        self.cursor = self.cnx.cursor(pymysql.cursors.DictCursor)
        self._set_max_user()
        self._set_max_productid()
        self._set_ratio_purchased()
        self.users = range(1, self.max_user+1)

    # returns the largest user index
    def _set_max_user(self):
        query = """select max(user_id) as max from insta2.combined """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        User.max_user = result[0]['max']

    # returns the largest product id
    def _set_max_productid(self):
        query = """select max(product_id) as max from insta2.combined """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        User.label_size = result[0]['max']

    # returns the average purchases per order
    def _set_ratio_purchased(self):
        query = "select count(*) as count from " + \
                "(select user_id, order_number from insta2.combined "+\
                " group by user_id, order_number) as t "
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        _order_count = result[0]['count']
        query = "select count(*) as count from insta2.combined"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        _total_count = result[0]['count']
        User.ratio_purchase = _total_count/_order_count

    # Returns all orders made by user on all purchases. It groups the purchases done in a single purchase into
    # a single vector. The output will be in the form (purchase_count * pred_size * 1)
    def get_data(self, user_id):
        query = "SELECT * FROM insta2.combined where user_id = " + str(user_id) + " order by order_number asc, " \
                                                                            "add_to_cart_order asc "
        self.cursor.execute(query)
        predictors = np.zeros((0, 32+User.max_user))
        labels = np.zeros((0, User.label_size, 2))

        order_number = -1
        label = User.__new_label()
        result = self.cursor.fetchall()
        for x in result:
            if order_number != x['order_number']:
                if order_number != -1:
                    labels = np.append(labels, label.reshape((1,  User.label_size, 2)), 0)
                order_number = x['order_number']
                order = User.__get_single(x)
                user_vec = User.__long_2_vec(user_id, self.max_user)
                order = np.append(order, user_vec, 1)
                predictors = np.append(predictors, order, 0)
                label = User.__new_label()
            label = User.__add_label(x, label)
        labels = np.append(labels, label.reshape((1, User.label_size, 2)), 0)

        return predictors, labels

    # creates a new empty data point
    @staticmethod
    def __new_label():
        label_ones = np.ones((User.label_size, 1))
        label_zeros = np.ones((User.label_size, 1))*0.00001
        output = np.stack((label_zeros, label_ones), 1).reshape((User.label_size, 2))
        return output

    # adds a new purchased item to the data point
    @staticmethod
    def __add_label(data, label):
        label[data['product_id']-1, 0] = 1
        label[data['product_id']-1, 1] = 0.00001
        return label

    # returns the portion of the data point that is not associated with the order
    @staticmethod
    def __get_single(data):
        output = np.zeros((1, 32))
        output[0, 0:7] = User.__long_2_vec(data['order_dow']+1, 7)
        output[0, 8:32] = User.__long_2_vec(data['order_hour_of_day'] + 1, 24)
        output[0, 7] = ((data['days_since_prior_order']-10.6659)/82.46)
        return output

    # generates the one hot vector
    @staticmethod
    def __long_2_vec(input, max):
        temp_arr = [0.000001] * (input-1)
        temp_arr += [1]
        temp_arr += [0.000001] * (max - input)
        output = np.array(temp_arr).reshape(1, max)
        return output
