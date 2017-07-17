lines = open('./List.txt', 'r').readlines()
names = [line.strip().split(' ')[0] for line in lines]
labels = [int(line.strip().split(' ')[1]) for line in lines]

temp = [0] * 10

output = open('./new_list.txt', 'w')
for i in range(len(lines)):
    temp[labels[i]] = 1
    line = names[i] + ' ' + ' '.join([str(l) for l in temp]) + '\n'
    temp[labels[i]] = 0
    output.write(line)
