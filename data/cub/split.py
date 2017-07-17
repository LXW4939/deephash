import codecs
import sys
import random
import numpy
import os

list_path = "./List.txt"

list_file = open(list_path, 'r')
lines = list_file.readlines()           

class_num = 10
unseen_cls_id = 9
train_num = 10000
query_num = 1000

random.shuffle(lines)
labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]
seen = {i: lines[i] for i in range(len(lines)) if labels[i][unseen_cls_id] != 1}
unseen = {i: lines[i] for i in range(len(lines)) if labels[i][unseen_cls_id] == 1}

train = [lines[index] for index in seen.keys()[:train_num]]
test = [lines[index] for index in unseen.keys()[:query_num]]
database = [lines[index] for index in seen.keys() + unseen.keys()[query_num:]]

train_path = "./train.txt"
test_path = "./test.txt"
database_path = "./database.txt"

train_file = open(train_path, 'w')
test_file = open(test_path, "w")
database_file = open(database_path, "w")

random.shuffle(train)
for i in range(train_num):
    train_file.write(train[i])
random.shuffle(test)
for item in test:
    test_file.write(item)
random.shuffle(database)
for item in database:
    database_file.write(item)
