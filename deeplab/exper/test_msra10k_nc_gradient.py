import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#net = caffe.Net('msra10k/config/deeplab_vgg16/trainkmeans_train.prototxt', 'msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel', caffe.TRAIN)
solver = caffe.SGDSolver('msra10k/config/deeplab_vgg16/solvernc_train.prototxt')
#solver.net.copy_from('msra10k/model/deeplab_vgg16/init.caffemodel')
solver.net.copy_from('msra10k/model/deeplab_vgg16/train_iter_1000.caffemodel')

imgid=45
for imgid in range(imgid):
    solver.net.forward()
    
training_net = solver.net
print training_net.blobs['nc_loss'].data.shape
print training_net.blobs['nc_loss'].data
print training_net.blobs['fc8_interp_softmax'].data.shape

prob = training_net.blobs['fc8_interp_softmax'].data
scores = training_net.blobs['fc8_interp']

print type(prob)
print prob[0,0,...,...].shape
print prob[0,0,...,...].dtype

data_dim = training_net.blobs['data_dim'].data
img_h = data_dim[0,0,0,0]
img_w = data_dim[0,0,0,1]
print 'img_w', img_w
print 'img_h', img_h


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
img=img[0:img_h,0:img_w,...]
plt.imshow(img, cmap='gray')
plt.axis("off")
print np.amax(img[...,...,0])
print np.amin(img[...,...,0])
plt.show(block=False)

mpimg.imsave("pythonoutput/"+str(imgid)+"input.png", img)


print np.amax(prob[0,0,...,...])
print np.amin(prob[0,0,...,...])
print np.amax(prob[0,1,...,...])
print np.amin(prob[0,1,...,...])
plt.figure(2)
plt.imshow(prob[0,0,0:img_h,0:img_w],  vmin=0, vmax=1)
plt.axis("off")
#plt.colorbar()
plt.show(block=False)
#plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.show()
#plt.savefig("pythonoutput/"+str(imgid)+"prob_0.png",bbox_inches='tight')

cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=1)

# map the normalized data to colors
# image is now RGBA (512x512x4) 

# save the image
plt.imsave("pythonoutput/"+str(imgid)+"prob_0.png", cmap(norm(prob[0,0,0:img_h,0:img_w])))



solver.net.backward()
plt.figure(3)
plt.imshow(prob[0,1,0:img_h,0:img_w],  vmin=0, vmax=1)
plt.axis("off")
#plt.colorbar()
plt.show(block=False)
#plt.imshow(prob[0,1,...,...], cmap='gray');
#plt.show()
#plt.savefig("pythonoutput/"+str(imgid)+"prob_1.png",bbox_inches='tight')

cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=1)
plt.imsave("pythonoutput/"+str(imgid)+"prob_1.png", cmap(norm(prob[0,1,0:img_h,0:img_w])))


scores_diff = scores.diff
print scores_diff.shape
print scores_diff.dtype

print np.amax(scores_diff[0,1,...,...])
print np.amin(scores_diff[0,1,...,...])

plt.figure(4)
im=plt.imshow(scores_diff[0,0,0:img_h,0:img_w]*1000,cmap="hot")
plt.axis("off")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="4%", pad=0.07)
plt.colorbar(im, cax=cax)
plt.show(block=False)
plt.savefig("pythonoutput/"+str(imgid)+"scores_diff_0.png",bbox_inches='tight')



plt.figure(5)
im=plt.imshow(scores_diff[0,1,0:img_h,0:img_w]*1000,cmap="hot")
plt.axis("off")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="4%", pad=0.07)
plt.colorbar(im, cax=cax)
plt.show(block=False)
plt.savefig("pythonoutput/"+str(imgid)+"scores_diff_1.png",bbox_inches='tight')


print "prob sum channel 0: ", np.sum(prob[0,0,...,...])
print "prob sum channel 1: ", np.sum(prob[0,1,...,...])
prob0 = prob[0,0,...,...]
print np.sum(np.multiply(prob0, imgblob[0,...,...]))
print np.sum(np.multiply(prob0, np.absolute(imgblob[0,...,...])))
#exit()

prob_diff = training_net.blobs['fc8_interp_softmax'].diff
print prob_diff.shape
print prob_diff.dtype

print np.amax(prob_diff[0,1,0:img_h,0:img_w])
print np.amin(prob_diff[0,1,0:img_h,0:img_w])

plt.figure(6)
im=plt.imshow(prob_diff[0,0,0:img_h,0:img_w]*1000,cmap="hot")
plt.axis("off")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="4%", pad=0.07)
plt.colorbar(im, cax=cax)
plt.show(block=False)
plt.savefig("pythonoutput/"+str(imgid)+"prob_diff_0.png",bbox_inches='tight')

plt.figure(7)
im=plt.imshow(prob_diff[0,1,0:img_h,0:img_w]*1000,cmap="hot")
plt.axis("off")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="4%", pad=0.07)
plt.colorbar(im, cax=cax)
plt.show(block=False)
plt.savefig("pythonoutput/"+str(imgid)+"prob_diff_1.png",bbox_inches='tight')

