import numpy as np
import random

with open('images.txt') as fin:
    img_list = fin.readlines()
seen = np.loadtxt('ids.txt')

prefix = "/home/dataset/CUB_200_2011/images/"

ftr = open('train.txt', 'w')
fte = open('test.txt', 'w')
fdb = open('database.txt', 'w')
funseen = open('unseen.txt', 'w')

random.shuffle(img_list)
label_dim = 200
query_num = 1000
n_query = 0
for img in img_list:
    img_path = img.split()[1]
    cls = int(img_path[:3])-1
    cls_str = ' '.join([str(1 * (i == cls)) for i in range(label_dim)])
    to_write = '{}{} {}\n'.format(prefix, img_path, cls_str)
    if seen[cls] or n_query >= query_num:
        ftr.write(to_write)
    else:
        fte.write(to_write)
        n_query = n_query + 1
    fdb.write(to_write)
    if not seen[cls]:
        funseen.write(to_write)

ftr.close()
fte.close()
