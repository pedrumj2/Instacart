class CrossEntropy(object):
    def __init__(self):
        self.current = 0
        self.max = 5
        self.values = list()

    def add_value(self, new_cross_entropy):
        if self.current == self.max:
            self.values[0:self.max-1] = self.values[1:self.max]
            self.values[self.max-1] = new_cross_entropy
        else:
            self.values.append(new_cross_entropy)
            self.current += 1

    def print_res(self):
        sum = 0
        count =0
        for i in range(self.current):
            sum += self.values[i]
            count += 1.0
        print(sum/count)

