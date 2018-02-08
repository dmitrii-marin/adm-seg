import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('msra10k/config/deeplab_vgg16/trainkmeans_train.prototxt', 'msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel', caffe.TRAIN)
solver = caffe.SGDSolver('msra10k/config/deeplab_vgg16/solverkmeans_train.prototxt')
solver.net.copy_from('msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel')
training_net = solver.net

solver.net.forward()

print training_net.blobs['kmeans_loss'].data.shape
print training_net.blobs['kmeans_loss'].data
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
img[...,...,2] = imgblob[0,...,...] + 104.008
img[...,...,1] = imgblob[1,...,...] + 116.669
img[...,...,0] = imgblob[2,...,...] + 122.675
print img.dtype
print img.shape
plt.figure(1)
plt.imshow(img, cmap='gray')
print np.amax(img[...,...,0])
print np.amin(img[...,...,0])
plt.show(block=False)

print np.amax(prob[0,0,...,...])
print np.amin(prob[0,0,...,...])
print np.amax(prob[0,1,...,...])
print np.amin(prob[0,1,...,...])
plt.figure(2)
plt.imshow(prob[0,0,...,...], cmap='gray');
#plt.colorbar()
plt.show(block=False)
#plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.show()

solver.net.backward()
plt.figure(3)
plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.colorbar()
plt.show(block=False)
#plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.show()

scores_diff = scores.diff
print scores_diff.shape
print scores_diff.dtype

print np.amax(scores_diff[0,1,...,...])
print np.amin(scores_diff[0,1,...,...])

plt.figure(4)
plt.imshow(scores_diff[0,0,...,...],cmap="hot");
plt.colorbar()
plt.show(block=False)

plt.figure(5)
plt.imshow(scores_diff[0,1,...,...],cmap="hot");
plt.colorbar()
plt.show(block=False)


print "prob sum channel 0: ", np.sum(prob[0,0,...,...])
print "prob sum channel 1: ", np.sum(prob[0,1,...,...])
prob0 = prob[0,0,...,...]
print np.sum(np.multiply(prob0, imgblob[0,...,...]))
print np.sum(np.multiply(prob0, np.absolute(imgblob[0,...,...])))
#exit()

prob_diff = training_net.blobs['fc8_interp_softmax'].diff
print prob_diff.shape
print prob_diff.dtype

print np.amax(prob_diff[0,1,...,...])
print np.amin(prob_diff[0,1,...,...])

plt.figure(6)
plt.imshow(prob_diff[0,0,...,...],cmap="hot");
plt.colorbar()
plt.show(block=False)

plt.figure(7)
plt.imshow(prob_diff[0,1,...,...],cmap="hot");
plt.colorbar()

plt.show()
