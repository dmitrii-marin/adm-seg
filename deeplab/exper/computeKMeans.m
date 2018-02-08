function [ kmeansenergy ] = computeKMeans( image, segmentation )
%COMPUTEKMEANS Summary of this function goes here
%   Detailed explanation goes here
% image: H*W*3
% segmentation: W*H*channels
[H W C] = size(image);
channels = size(segmentation, 3);
kmeansenergy = 0;
X = zeros(H*W, C);
for c=1:C
    X(:, c) = reshape(image(:,:,c), [H*W 1]);
end
for c=1:channels
    segmentation_c = segmentation(:,:,c);
    segmentation_c = segmentation_c(:);
    segmentation_size = sum(segmentation_c);
    color_sum = X .* repmat(segmentation_c, 1, C);
    color_sum = sum(color_sum);
    mean_color = color_sum / (segmentation_size + realmin);
    
    kmeansenergy = kmeansenergy + sum( sum((X - repmat(mean_color, H*W, 1)).^2,2).*segmentation_c);
end

kmeansenergy = kmeansenergy / 255 / 255;

end

