class Results(object):
    def __init__(self):
        self.TP = 0.0
        self.TN = 0.0
        self.FN = 0.0
        self.FP = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.F1 = 0.0

    def add_results(self, labels, results):
        for i in range(len(labels)):
            if labels[i] <= 0.5 and results[i] <= 0.5:
                self.TN += 1.0
            elif labels[i] <= 0.5 and results[i] >= 0.5:
                self.FP += 1.0
            elif labels[i] >= 0.5 and results[i] <= 0.5:
                self.FN += 1.0
            else:
                self.TP += 1.0
        self.__update()

    def __update(self):
        if (self.TP + self.FP) == 0:
            self.precision = 0.0
        else:
            self.precision = self.TP/(self.TP + self.FP)
        if (self.TP + self.FN) == 0:
            self.recall = 0.0
        else:
            self.recall = self.TP/(self.TP + self.FN)
        if (self.precision + self.recall) == 0:
            self.F1 = 0.0
        else:
            self.F1 = 2.0*(self.precision*self.recall)/(self.precision+self.recall)

    def print_output(self):
        print("TP, FP, TN, FN: %d, %d, %d, %d"%(
            self.TP, self.FP, self.TN, self.FN))
        print("precision:%f"%(
            self.precision))
        print("recall:%f"%(
            self.recall))
        print("F1:%f"%(
            self.F1))

