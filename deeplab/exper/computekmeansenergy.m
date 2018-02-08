function [ kmeansenergy ] = computekmeansenergy( img, segmentation, xyscale )
%COMPUTEKMEANSENERGY Summary of this function goes here
%   Detailed explanation goes here
kmeansenergy = 0;

[H W C] = size(img);
img_r = double(img(:,:,1));
img_g = double(img(:,:,2));
img_b = double(img(:,:,3));
img_x = repmat(1:W, H, 1);
img_y = repmat((1:H)',1,W);
data = double([img_r(:) img_g(:) img_b(:) img_x(:) * xyscale img_y(:) * xyscale]);

labels = unique(segmentation);

for k = numel(labels)
    i = labels(k);
    data_i = double(data(segmentation(:)==i,:));
    clustermean = mean(data_i);
    diff = data_i - repmat(clustermean, size(data_i,1), 1);
    kmeansenergy = kmeansenergy + sum(sum(diff.^2));
end

end

