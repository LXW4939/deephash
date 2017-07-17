import codecs
import sys
import random
import numpy
import os

list_path = "./train.txt"
train_path = "./train_semi.txt"

list_file = open(list_path, 'r')
lines = list_file.readlines()
print len(lines)

train_num = 1000
class_num = 81
n_duplicate_seen_data = 9

random.shuffle(lines)
train = lines[:train_num] * n_duplicate_seen_data
all_zero_label = " ".join(["0"] * class_num)
train += ["{} {}\n".format(line.strip().split(" ")[0], all_zero_label) for line in lines[train_num:]]

train_num = len(train)
print train_num
train_file = open(train_path, 'w')
random.shuffle(train)
for i in range(train_num):
    train_file.write(train[i])
