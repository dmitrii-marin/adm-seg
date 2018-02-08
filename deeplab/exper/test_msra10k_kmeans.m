close all
clear all
whichset='val';
fileID = fopen(['msra10k/list/' whichset '_id.txt']);
imgnames = textscan(fileID,'%s');
imgnames = imgnames{1};
fclose(fileID);
initialization_energies = [];
network_energies = [];
kmeans_energies = [];
for imgid = 26:26
    
imgname = imgnames{imgid};
img = imread(['~/Disney/data/MSRA10K/Imgs/' imgname '.jpg']);
[H W C] = size(img);

largeimg = zeros(513,513,3,'uint8');
largeimg(1:H, 1:W, :) = img;
%figure,imshow(img);
%axis on

% initialization
load(['msra10k/features/deeplab_vgg16/' whichset '/fc8/' imgname '_blob_0.mat']); % data is of WxHxC
probmap = data(1:(min(W,size(data,1))), 1:(min(H,size(data,2))),:);
probmap = canonicalForm( probmap );
[~,a] = max(probmap,[],3);
%figure,imagesc(a);
figure,subplot(1,3,1);
imshow(encodesegmentation( img, a ));title('initialization');
initialization_energy = computeKMeans( img, probmap );
initialization_energies = [initialization_energies initialization_energy];
imwrite(encodesegmentation( img, a ),['kmeansnetwork/' imgname '_init.png']);

% kmeans algorithm
[ kmeanssegmentation, kmeansiterativeenergies ] = runKmeans( img, probmap);
[~,a] = max(kmeanssegmentation,[],3);
subplot(1,3,2);imshow(encodesegmentation( img, a ));title('KMeans algorithm');
kmeans_energy = computeKMeans(img, kmeanssegmentation);
kmeans_energies = [kmeans_energies kmeans_energy];
imwrite(encodesegmentation( img, a ),['kmeansnetwork/' imgname '_kmeans.png']);


% kmeans network
load(['msra10k/featureskmeans/deeplab_vgg16/' whichset '/fc8/' imgname '_blob_0.mat']); % data is of WxHxC
probmap = data(1:(min(W,size(data,1))), 1:(min(H,size(data,2))),:);
probmap = canonicalForm( probmap );
[~,a] = max(probmap,[],3);
%figure,imagesc(a);
subplot(1,3,3);imshow(encodesegmentation( img, a ));title('KMeans network');
network_energy = computeKMeans( img, probmap );
network_energies = [network_energies network_energy];
imwrite(encodesegmentation( img, a ),['kmeansnetwork/' imgname '_network.png']);

%figure,plot(kmeansiterativeenergies);title('Kmeans iterations');

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

disp(['mean initialization energies: ' num2str(mean(initialization_energies))]);
disp(['mean kmeans energies: ' num2str(mean(kmeans_energies))]);
disp(['mean network energies: ' num2str(mean(network_energies))]);
