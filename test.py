"""
"""
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import glob
import os


interval = 2
seq_length = 2
output_dim = 3

filename = glob.glob("* - processed.csv")[0]
json_filename = filename + " " + str(interval) + " " + str(seq_length) + " " + ".json"
with open("DATA/seqed data/" + json_filename, 'rt') as fp:
    data = json.load(fp)
    cnt_num_label = data[0]
    cnt_label = [0 for _ in range(output_dim)]
    input = np.array(data[1])
    label = np.array(data[2])
    fp.close()

posi = np.delete(label, np.s_[0], axis=1)  # + binary

for i in range(len(posi)):
    if (posi[i] == np.array([0, 0])).all():
        posi[i] = np.array([1, 0])

print(posi.tolist())

print("=================")
nega = np.delete(label, np.s_[2], axis=1)  # + binary
for i in range(len(nega)):
    if (nega[i] == np.array([0, 0])).all():
        nega[i] = np.array([0, 1])

print(nega.tolist())

