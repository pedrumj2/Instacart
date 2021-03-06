import sys

from DataMod.DataSets import DataSets
from DataMod.RnnAux import RnnAux
from DataMod.User import User
from Models import Model1

# Command line arguements.
# args[1]: db password
# args[2]: db ip
# args[3]: db name
# args[4]: hidden layer count
# args[5]: rnn lenght
# args[6]: ratio of data points to use
# args[7]: ratio of data points select by args[6] to be put aside for validation
# args[8]: number of iterations to run on training the data
users = User(sys.argv[1], sys.argv[2], sys.argv[3], "root")
dataSet = DataSets(int(sys.argv[6]), float(sys.argv[7]), users)
hidden_layer = int(sys.argv[4])
rnn_length = int(sys.argv[5])
rnnAux = RnnAux(dataSet, rnn_length)
count_train = int(sys.argv[8])
Model1.run(rnnAux, rnn_length, count_train, hidden_layer)


