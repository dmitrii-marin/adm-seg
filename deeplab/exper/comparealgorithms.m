close all
clear all

% change this path if you install the VOC code elsewhere
addpath(['~/Disney/data/VOCdevkit/VOCcode']);
addpath('/home/mtang73/Disney/data/pascal_scribble');
% initialize VOC options
VOCinit;

cmap = VOClabelcolormap();

fileID = fopen('voc12/list/val_id.txt');
imgnames = textscan(fileID,'%s');
imgnames = imgnames{1};
fclose(fileID);

for imgid = 201:500
    imgname = imgnames{imgid};
    % original image
    img = imread(['~/Disney/data/VOCdevkit/VOC2012/JPEGImages/' imgname '.jpg']);
    [H W C] = size(img);
    %figure,imshow(img);
    %axis on
    imwrite(img,['compare/' imgname '.png']);
    % ground truth
    gt = imread(['~/Disney/data/VOCdevkit/VOC2012/SegmentationClassAug/' imgname '.png']);
    imwrite(encodemask( img, gt, cmap ), ['compare/' imgname '_gt.png']);
    % full supervision
    load(['voc12/features/deeplab_msc_largeFOV/val/crf/' imgname '_blob_0.mat']); % data is of WxHxC
    [~,a] = max(data,[],3);
    a = a';
    a = a(1:H,1:W);
    %figure,imagesc(a);title('full');
    imwrite(encodemask( img, a-1, cmap ), ['compare/' imgname '_full.png']);
    % our weak supervision
    load(['pascal_scribble/features/deeplab_msc_largeFOV/val/crf/' imgname '_blob_0.mat']); % data is of WxHxC
    [~,a] = max(data,[],3);
    a = a';
    a = a(1:H,1:W);
    %figure,imagesc(a);title('our weak');
    imwrite(encodemask( img, a-1, cmap ), ['compare/' imgname '_ourweak.png']);
    
    intmethods = {'grabcut','normalizedcut','kernelcut'};
    for m=1:3
    % grabcut
        load(['pascal_scribble_int/features/deeplab_msc_largeFOV/val/crf' intmethods{m}...
            '/' imgname '_blob_0.mat']); % data is of WxHxC
        [~,a] = max(data,[],3);
        a = a';
        a = a(1:H,1:W);
        %figure,imagesc(a);title(intmethods{m});
        imwrite(encodemask( img, a-1, cmap ), ['compare/' imgname '_' intmethods{m} '.png']);
    end


end
