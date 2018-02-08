from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#training_net = caffe.Net('pascal_scribble/config/deeplab_largeFOV/trainwithncloss_train.prototxt', 'pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel', caffe.TRAIN)
#training_net = caffe.Net('pascal_scribble/config/deeplab_largeFOV/train_train.prototxt', 'pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel', caffe.TRAIN)

solver = caffe.SGDSolver('msra10k/config/deeplab_vgg16/solver_train.prototxt')
solver.net.copy_from('msra10k/model/deeplab_vgg16/train_iter_2000.caffemodel')
training_net = solver.net

solver.step(1)
imgid=4

# RGB image
imgblob = training_net.blobs['data'].data
imgblob = imgblob[imgid,...,...,...]
(C, H, W) = imgblob.shape
img = np.zeros((H, W, C), dtype=np.uint8);
print '0 0 color ', imgblob[...,0,0]

img[...,...,2] = imgblob[0,...,...] + 104.008
img[...,...,1] = imgblob[1,...,...] + 116.669
img[...,...,0] = imgblob[2,...,...] + 122.675
print img.dtype
print img.shape

fc8_msra10k_data=training_net.blobs['fc8_msra10k'].data
fc8_msra10k_data=fc8_msra10k_data[imgid,...,...,...]
fc8_msra10k_diff=training_net.blobs['fc8_msra10k'].diff
fc8_msra10k_diff=fc8_msra10k_diff[imgid,...,...,...]
#fc8_interp_sum = fc8_interp_data[0,0,...,...] + fc8_interp_data[0,1,...,...]
#plt.imshow(fc8_interp_sum, vmin=0, vmax=1);
#plt.colorbar()
#plt.savefig('pythonoutput/' + str(i) + 'prob' + '.png')
#plt.show(block=False)

fig=plt.figure()
for i in range(2):
    plt.subplot(1,2,i)
    plt.imshow(fc8_msra10k_data[i,...,...]);
    plt.colorbar()
    #plt.savefig('pythonoutput/' + str(i) + 'prob' + '.png')
    plt.show(block=False)
    plt.title('Input data channel '+`i`);
    #plt.close(fig)
    
fig=plt.figure()
for i in range(2):
    plt.subplot(1,2,i)
    plt.imshow(fc8_msra10k_diff[i,...,...]);
    plt.title('fc8_msra10k_diff channel '+`i`);
    plt.colorbar()
    #plt.savefig('pythonoutput/' + str(i) + 'prob' + '.png')
    plt.show(block=False)
    #plt.close(fig)

plt.figure()
plt.imshow(img, cmap='gray')
print np.amax(img[...,...,0])
print np.amin(img[...,...,0])
plt.show(block=True)
    

exit()

print training_net.blobs['fc8_interp_softmax'].data.shape


prob = training_net.blobs['fc8_interp_softmax'].data
scores = training_net.blobs['fc8_interp']


print type(prob)
print prob[0,0,...,...].shape
print prob[0,0,...,...].dtype

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
