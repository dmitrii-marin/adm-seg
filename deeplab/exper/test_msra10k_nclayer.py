import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('msra10k/config/deeplab_vgg16/trainkmeans_train.prototxt', 'msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel', caffe.TRAIN)
solver = caffe.SGDSolver('msra10k/config/deeplab_vgg16/solvernclayer_train.prototxt')
solver.net.copy_from('msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel')
training_net = solver.net

solver.net.forward()

print training_net.blobs['nc_loss'].data.shape
print training_net.blobs['nc_loss'].data
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
plt.axis('off')
plt.title("Input");
print np.amax(img[...,...,0])
print np.amin(img[...,...,0])
plt.show(block=False)

print np.amax(prob[0,0,...,...])
print np.amin(prob[0,0,...,...])
print np.amax(prob[0,1,...,...])
print np.amin(prob[0,1,...,...])
plt.figure(2)
plt.imshow(prob[0,0,...,...], cmap='gray');
plt.title("unary channel 0")
#plt.colorbar()
plt.show(block=False)
#plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.show()

solver.net.backward()
plt.figure(3)
plt.imshow(prob[0,1,...,...], cmap='gray');
plt.title("unary channel 1")
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
plt.title("score gradient channel 0");
plt.show(block=False)

plt.figure(5)
plt.imshow(scores_diff[0,1,...,...],cmap="hot");
plt.title("score gradient channel 1");
plt.colorbar()
plt.show(block=False)


print "prob sum channel 0: ", np.sum(prob[0,0,...,...])
print "prob sum channel 1: ", np.sum(prob[0,1,...,...])
prob0 = prob[0,0,...,...]
print np.sum(np.multiply(prob0, imgblob[0,...,...]))
print np.sum(np.multiply(prob0, np.absolute(imgblob[0,...,...])))
#exit()


nc_bound = training_net.blobs['nc_bound'].data

plt.figure(6)
plt.imshow(nc_bound[0,0,...,...],cmap="hot");
plt.colorbar()
plt.title("nc bound channel 0");
plt.show(block=False)

plt.figure(7)
plt.imshow(nc_bound[0,1,...,...],cmap="hot");
plt.colorbar()
plt.title("nc bound channel 1");
plt.show(block=False)

logp_nc_bound = training_net.blobs['logp_nc_bound'].data

plt.figure(8)
plt.imshow(logp_nc_bound [0,0,...,...],cmap="hot");
plt.colorbar()
plt.title("nc bound scaled channel 0");
plt.show(block=False)

plt.figure(9)
plt.imshow(logp_nc_bound [0,1,...,...],cmap="hot");
plt.colorbar()
plt.title("nc bound scaled channel 1");
plt.show(block=False)

bound_softmax = training_net.blobs['bound_softmax'].data

plt.figure(10)
plt.imshow(bound_softmax[0,0,...,...],cmap="gray");
plt.colorbar()
plt.title("bound softmax channel 0");
plt.show(block=False)

plt.figure(11)
plt.imshow(bound_softmax[0,1,...,...],cmap="gray");
plt.colorbar()
plt.title("bound softmax channel 1");
plt.show(block=False)

plt.figure(12)
plt.imshow(prob[0,0,...,...]>prob[0,1,...,...]);
plt.title("unary segmentation")
plt.axis('off')
plt.show(block=False)

plt.figure(13)
plt.imshow(bound_softmax[0,0,...,...]>bound_softmax[0,1,...,...]);
plt.title("unary + NC iteration 1")
plt.axis('off')
plt.show(block=False)

bound_softmax2 = training_net.blobs['bound_softmax2'].data
plt.figure(14)
plt.imshow(bound_softmax2[0,0,...,...]>bound_softmax2[0,1,...,...]);
plt.title("unary + NC iteration 2")
plt.axis('off')
plt.show()
