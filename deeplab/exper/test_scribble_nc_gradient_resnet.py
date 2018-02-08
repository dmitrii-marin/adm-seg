from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('pascal_scribble/config/resnet-101/solverwithncloss_train.prototxt')
solver.net.copy_from('pascal_scribble/model/resnet-101/train_iter_20000.caffemodel')
training_net = solver.net

solver.step(1)

print 'nc loss', training_net.blobs['nc_loss'].data
print training_net.blobs['fc8_interp_softmax'].data.shape


prob = training_net.blobs['fc8_interp_softmax'].data
scores = training_net.blobs['fc8_interp']


print type(prob)
print prob[0,0,...,...].shape
print prob[0,0,...,...].dtype

# RGB image
imgblob = training_net.blobs['data'].data
imgblob = imgblob[0,...,...,...]
(C, H, W) = imgblob.shape
img = np.zeros((H, W, C), dtype=np.uint8);
print '0 0 color ', imgblob[...,0,0]

img[...,...,2] = imgblob[0,...,...] + 104.008
img[...,...,1] = imgblob[1,...,...] + 116.669
img[...,...,0] = imgblob[2,...,...] + 122.675
print img.dtype
print img.shape
plt.figure()
plt.imshow(img, cmap='gray')
print np.amax(img[...,...,0])
print np.amin(img[...,...,0])
plt.show(block=False)

scores_diff = scores.diff
print scores_diff.shape
print scores_diff.dtype

print np.amax(scores_diff[0,1,...,...])
print np.amin(scores_diff[0,1,...,...])


print "prob sum channel 0: ", np.sum(prob[0,0,...,...])
print "prob sum channel 1: ", np.sum(prob[0,1,...,...])
prob0 = prob[0,0,...,...]
print np.sum(np.multiply(prob0, imgblob[0,...,...]))
print np.sum(np.multiply(prob0, np.absolute(imgblob[0,...,...])))
#exit()

prob_diff = training_net.blobs['fc8_interp_softmax'].diff

shrink_scores_diff = training_net.blobs['fc8_interp'].diff

temp=prob_diff[0,0:18:17,...,...]
print temp.shape

#minidx = prob_diff[0,...,...,...].argmin(axis=0)
#print minidx.shape
#fig=plt.figure()
#plt.imshow(minidx);
#plt.show(block=True)
#exit()

print np.amax(prob[0,0,...,...])
print np.amin(prob[0,0,...,...])
print np.amax(prob[0,1,...,...])
print np.amin(prob[0,1,...,...])
for i in range(21):
    fig=plt.figure()
    plt.imshow(prob[0,i,...,...], vmin=0, vmax=1);
    plt.colorbar()
    plt.savefig('pythonoutput/' + str(i) + 'prob' + '.png')
    plt.show(block=False)
    plt.close(fig)




for i in range(21):
    fig = plt.figure()
    plt.imshow(scores_diff[0,i,...,...],cmap="hot");
    plt.colorbar()
    plt.savefig('pythonoutput/' + str(i) + 'scorediff' + '.png')
    plt.show(block=False)
    plt.close(fig)
    
#for i in range(21):
#    fig = plt.figure()
#    plt.imshow(shrink_scores_diff[0,i,...,...],cmap="hot");
#    plt.colorbar()
#    plt.savefig('pythonoutput/' + str(i) + 'shrinkscorediff' + '.png')
#    plt.show(block=False)
#    plt.close(fig)
    
for i in range(21):
    fig = plt.figure()
    plt.imshow(prob_diff[0,i,...,...],cmap="hot");
    plt.colorbar()
    plt.savefig('pythonoutput/' + str(i) + 'probdiff' + '.png')
    plt.show(block=False)
    plt.close(fig)
    
labels = training_net.blobs['label'].data
plt.figure()
plt.imshow(labels[0,0,...,...])
plt.show()
