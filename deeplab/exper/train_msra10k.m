addpath ../code/matlab
model = 'msra10k/config/deeplab_vgg16/trainkmeans_train.prototxt';
weights = 'msra10k/model/deeplab_vgg16/train_iter_20000.caffemodel';
caffe.set_mode_gpu();
caffe.set_device(0);
% create net and load weights
net = caffe.Net(model, weights, 'train'); 