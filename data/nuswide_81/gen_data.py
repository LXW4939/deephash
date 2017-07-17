import numpy as np
import scipy as sp
import os
import cPickle
from skimage import io
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
def showImage( batch_id, dictionary, imSize, attr, outfile):
	images = dictionary.get('data')
	labels = dictionary.get('labels')
	for i in xrange(10000):
		singleImage = images[i]

		recon = np.zeros( (imSize, imSize, 3), dtype = np.uint8 )
		singleImage = singleImage.reshape( (imSize*3, imSize))

		red = singleImage[0:imSize,:]
		blue = singleImage[imSize:2*imSize,:]
		green = singleImage[2*imSize:3*imSize,:]

		recon[:,:,0] = red
		recon[:,:,1] = blue
		recon[:,:,2] = green

		outpath = os.path.abspath(".") + "/" + attr + "/" + str(batch_id) + "_" + str(i) + ".jpg"
		#recon = resize(recon, (256, 256))
		io.imsave(outpath, recon)
		outfile.write(outpath + " " + str(labels[i]) + "\n")

if __name__ == "__main__":
	train_list = open("train.txt", "w")
	test_list = open("test.txt", "w")
	for i in xrange(5):
		print 'batch ', i
		fo = open("./cifar-10-batches-py/data_batch_%s" % str(i + 1), "rb")
		dictionary = cPickle.load(fo)
		fo.close()
		showImage(i, dictionary, 32, "train", train_list)
	print 'test batch'
	fo = open("./cifar-10-batches-py/test_batch", "rb")
	dictionary = cPickle.load(fo)
	fo.close()
	showImage(999, dictionary, 32, "test", test_list)
	train_list.close()
	test_list.close()


