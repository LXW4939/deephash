import sys, time
from multiprocessing import Pool
import numpy as np

class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self, s=None):
        self.count += 1
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        if s:
            print s
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}:\t'.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
        return


def get_allrel(args):
    return _get_allrel(*args)

def _get_allrel(dis, database_codes, query_codes, offset):
    print 'getting relations of: ', offset
    all_rel = np.zeros([query_codes.shape[0], database_codes.shape[0]])
    for i in xrange(query_codes.shape[0]):
        for j in xrange(database_codes.shape[0]):
            A1 = np.where(query_codes[i] == 1)[0]
            A2 = np.where(database_codes[j] == 1)[0]
            all_rel[i, j] = sum([dis[x][y] for x in A1 for y in A2])
        if i % 100 == 0:
            print offset, ' reaching: ', i
    print "allrel part ", offset
    print all_rel
    print "query codes wrong:"
    print np.sum(np.sum(query_codes, 1) != 4)
    print "database codes wrong:"
    print np.sum(np.sum(database_codes, 1) != 4)
    return all_rel

class MAPs:
    def __init__(self, R):
        self.R = R

    def distance(self, a, b):
        return np.dot(a, b)

    def get_mAPs_by_feature(self, database, query):
        ips = np.dot(query.output, database.output.T)
        #norms = np.sqrt(np.dot(np.reshape(np.sum(query.output ** 2, 1), [query.n_samples, 1]), np.reshape(np.sum(database.output ** 2, 1), [1, database.n_samples])))
        #self.all_rel = ips / norms
        self.all_rel = ips
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        print "#calc mAPs# calculating mAPs"
        bar = ProgressBar(total=self.all_rel.shape[0])
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R+1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            bar.move()
        print "mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

class MAPs_CQ:
    def __init__(self, C, subspace_num, subcenter_num, R):
        self.C = C
        self.subspace_num = subspace_num
        self.subcenter_num = subcenter_num
        self.R = R


    def get_mAPs_SQD(self, database, query):
        self.all_rel = np.dot(np.dot(query.codes, self.C), np.dot(database.codes, self.C).T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        #print "#calc mAPs# calculating mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            #if i % 100 == 0:
                #print "step: ", i
        print "SQD mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

    def get_mAPs_AQD(self, database, query):
        self.all_rel = np.dot(query.output, np.dot(database.codes, self.C).T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        #print "#calc mAPs# calculating AQD mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            #if i % 100 == 0:
                #print "step: ", i
        print "AQD mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

    def get_mAPs_by_feature(self, database, query):
        self.all_rel = np.dot(query.output, database.output.T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.label
        database_labels = database.label
        #print "#calc mAPs# calculating mAPs"
        for i in xrange(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
            #if i % 100 == 0:
                #print "step: ", i
        print "Feature mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))

    def get_mAPs_cls(self, database, word_dict):
        self.all_rel = np.dot(word_dict, np.dot(database.codes, self.C).T)
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        database_labels = database.label
        L = self.all_rel.shape[0]
        for i in xrange(L):
            idx = ids[i, :]
            label = [1 if index == i else -1 for index in range(L)]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            else:
                APx.append(0.0)
        print "cls mAPs: ", np.mean(np.array(APx))
        return np.mean(np.array(APx))
