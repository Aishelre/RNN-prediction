"""
strength에 대한 전처리가 된 파일을
MinMaxScaler와 유사하게 처리하여
'processed.csv'에 저장한다.
"""
import numpy as np

filename = 'st strength.csv'

def processing(data):
    arr = np.empty((len(data)), float)
    for i in range(len(data)):
        if data[i] >= 0:
            numerator = data[i] - np.min(data)
            denominator = np.max(data) - np.min(data)
            arr[i] = numerator / (denominator + 1e-7)
        elif data[i] < 0:
            numerator = data[i] - np.max(data)
            denominator = np.max(data) - np.min(data)
            arr[i] = numerator / (denominator + 1e-7)

    return arr


def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

c=0

with open(filename, 'rt') as fp:
    with open('processed.csv', 'wt') as p:
        for line in fp:
            c+=1
            print(c)
            line = line.split(',')[:-1]
            line = list(map(float, line))

            p.write(str(line[0]) + ',' +  str(line[1]) + ',')
            data = np.array(line[2:])
            #data = MinMaxScaler(data)
            data = processing(data)
            for i in range(len(data)):
                p.write(str(data[i]) + ',')
            p.write('\n')