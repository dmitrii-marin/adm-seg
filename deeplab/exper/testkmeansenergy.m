close all
clear all
whichset='val';
fileID = fopen(['voc12/list/' whichset '_id.txt']);
imgnames = textscan(fileID,'%s');
imgnames = imgnames{1};
fclose(fileID);

for i=1:numel(imgnames)
    imgname = imgnames{i};
    img = imread(['~/Disney/data/VOCdevkit/VOC2012/JPEGImages/' imgname '.jpg']);
    %figure,imshow(img);
    [H W C] = size(img);
    load(['voc12/features/deeplab_vgg16/val/fc8/' imgname '_blob_0.mat']); % data is of WxHxC
    probmap = data(1:(min(W,size(data,1))), 1:(min(H,size(data,2))),:);
    probmap = canonicalForm( probmap );
    [~,segmentation] = max(probmap,[],3);
    segmentation = int32(segmentation);
    %figure,imagesc(segmentation);
    
    for k = 1:10
        kmeansenergies(i,k) = computekmeansenergy( img, segmentation, (k-1) * 0.5 );
    end
    disp(['image ' num2str(i)]);
end

meanenergy = mean(kmeansenergies);