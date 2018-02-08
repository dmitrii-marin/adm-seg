%close all
clear all
classes = {'bg', 'aeroplane','bicycle','bird','boat',...
                 'bottle','bus','car','cat',...
                 'chair','cow','diningtable','dog',...
                 'horse','motorbike','person','pottedplant',...
                 'sheep','sofa','train','tvmonitor'};
fileID = fopen('voc12/list/val_id.txt');
imgnames = textscan(fileID,'%s');
imgnames = imgnames{1};
fclose(fileID);
for imgid = 1:20
    
imgname = imgnames{imgid};
img = imread(['~/Disney/data/VOCdevkit/VOC2012/JPEGImages/' imgname '.jpg']);
%figure,imshow(img);
%axis on
load(['voc12/featuresnc3/deeplab_vgg16/val/fc8/' imgname '_blob_0.mat']); % data is of WxHxC
[~,a] = max(data,[],3);
a = a';
%figure,imagesc(a);

%encodedimg = imresize(img,size(a));
encodedimg = encodesegmentation(img, a);
figure,imshow(encodedimg);
continue;
[a] = max(data,[],3);
figure;
numlabels = 2;
for i=1:2
    probmap = data(:,:,i)';
    probmap = probmap(1:H,1:W);
    if numlabels==21
        subplot(3,7,i),imagesc(probmap);
        title([num2str(i) ' ' classes{i}]);axis off
    else
        subplot(1,numlabels,i),imagesc(probmap);
        axis equal
        axis off
    end
    colormap jet
    caxis([0 1])
    %colorbar
end

end
