import sys

from DataMod.DataSets import DataSets
from DataMod.RnnAux import RnnAux
from DataMod.User import User
from Models import Model1

users = User(sys.argv[1], sys.argv[2], sys.argv[3], "root")
dataSet = DataSets(float(sys.argv[6]), float(sys.argv[7]), users)
hidden_layer = int(sys.argv[4])
rnn_length = int(sys.argv[5])
rnnAux = RnnAux(dataSet, rnn_length)


count_train = int(sys.argv[8])
model_no = int(sys.argv[9])

if model_no == 1:
    Model1.run(rnnAux, rnn_length, count_train, hidden_layer)


