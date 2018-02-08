from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#training_net = caffe.Net('pascal_scribble/config/deeplab_largeFOV/trainwithncloss_train.prototxt', 'pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel', caffe.TRAIN)
#training_net = caffe.Net('pascal_scribble/config/deeplab_largeFOV/train_train.prototxt', 'pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel', caffe.TRAIN)

#solver = caffe.SGDSolver('pascal_scribble/config/deeplab_largeFOV/solverwithncloss_train.prototxt')

train_net = caffe.Net('train_box.prototxt', caffe.TRAIN)
#test_net.copy_from('voc12/model/deeplab_vgg16/train_iter_2000.caffemodel')
#test_net = solver.net

for i in range(5):
    train_net.forward()

data_dim = train_net.blobs['data_dim'].data
img_h = data_dim[0,0,0,0]
img_w = data_dim[0,0,0,1]
print img_h
print img_w
print data_dim

# RGB image
imgblob = train_net.blobs['data'].data
imgblob = imgblob[0,...,...,...]
(C, H, W) = imgblob.shape
img = np.zeros((img_h, img_w, C), dtype=np.uint8);
print '0 0 color ', imgblob[...,0,0]

img[...,...,2] = imgblob[0,0:img_h,0:img_w] + 104.008
img[...,...,1] = imgblob[1,0:img_h,0:img_w] + 116.669
img[...,...,0] = imgblob[2,0:img_h,0:img_w] + 122.675
print img.dtype
print img.shape
plt.figure(1)
fig=plt.imshow(img, cmap='gray')
#plt.axis("off")
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
plt.show(block=False)

# box labeling
box = train_net.blobs['label'].data
box = box[0,0,...,...]
box = box[0:img_h,0:img_w]
print box.shape
print np.amax(box)
print np.amin(box)

plt.figure(2)
fig=plt.imshow(box)
#plt.axis("off")
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)
plt.show(block=True)


exit(-1)

SegAccuracy = test_net.blobs['accuracy'].data
print SegAccuracy

# RGB image
imgblob = test_net.blobs['data'].data
imgblob = imgblob[0,...,...,...]
(C, H, W) = imgblob.shape
img = np.zeros((img_h, img_w, C), dtype=np.uint8);
print '0 0 color ', imgblob[...,0,0]

img[...,...,2] = imgblob[0,0:img_h,0:img_w] + 104.008
img[...,...,1] = imgblob[1,0:img_h,0:img_w] + 116.669
img[...,...,0] = imgblob[2,0:img_h,0:img_w] + 122.675
print img.dtype
print img.shape
plt.figure(1)
fig=plt.imshow(img, cmap='gray')
plt.axis("off")
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show(block=False)
plt.savefig("horse.png", bbox_inches='tight', pad_inches = 0)

print 'nc loss', test_net.blobs['nc_loss'].data
print test_net.blobs['fc8_interp_softmax'].data.shape


prob = test_net.blobs['fc8_interp_softmax'].data
prob = prob[0,...,...,...]
print prob.shape

segmentation = np.argmax(prob,axis=0)

print segmentation.shape
segmentation = segmentation[0:img_h,0:img_w]


plt.figure(2)
fig=plt.imshow(segmentation);
plt.axis("off")
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show(block=False)
plt.savefig("test.png", bbox_inches='tight', pad_inches = 0)



exit(-1)



scores = test_net.blobs['fc8_interp']


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

shrink_scores_diff = training_net.blobs['fc8_pascal_scribble'].diff

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
