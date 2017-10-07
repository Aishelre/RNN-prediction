import numpy as np
import glob
import json

class Set_dataset():
    def __init__(self):
        self.output_dim = 3
        self.seq = [1,2,3,4,5,6,7,8,9,10]
        self.int = [2,3,4,5]

        self.start()

    def start(self):
        csv_list = glob.glob("DATA/Trainable-file/* - processed.csv")
        print(csv_list)

        for filename in csv_list:
            for int in self.int:
                for seq in self.seq:
                    self.interval = int
                    self.seq_length = seq
                    self.set_dataset(filename)

    def set_dataset(self, filename):
        xy = np.loadtxt(filename, delimiter=',')  # , usecols=range(data_dim + 2))  # numpy.ndarray 타입
        self.data_dim = len(xy[0]) - 2

        self.cnt_num_label = [0 for _ in range(self.output_dim)]
        self.cnt_label = [0 for _ in range(self.output_dim)]

        x = xy[..., 2:]  # 시간, 현재 가격을 제외한 호가 항목
        y = xy[..., 1]  # 현재 가격*

        self.input = np.empty((1, self.seq_length, self.data_dim), float)
        self.label = np.empty((1, self.output_dim), int)

        print("label data generating - - -")
        for i in range(self.seq_length - 1, len(y) - self.interval):
            val = y[i + self.interval] - y[i]
            if val > 0:
                lb = np.array([0, 0, 1])  # +
                self.cnt_num_label[2] += 1
            elif val < 0:
                lb = np.array([1, 0, 0])  # -
                self.cnt_num_label[0] += 1
            else:
                lb = np.array([0, 1, 0])  # 0
                self.cnt_num_label[1] += 1
            self.label = np.vstack((self.label, lb))
        self.label = self.label[1:]  # np.empty에서 생성한 label[0]을 지운다.

        print("input data generating - - -")
        for i in range(0, len(y) - self.seq_length - self.interval + 1):
            input_ = np.array([x[i:i + self.seq_length]])
            self.input = np.vstack((self.input, input_))
        self.input = self.input[1:]

        with open("DATA/seqed data/"+ filename[20:] + " " + str(self.interval) + " " + str(self.seq_length) + " " + ".json", 'wt') as fp:
            json.dump([self.cnt_num_label, self.input.tolist(), self.label.tolist()], fp)

Set_dataset()