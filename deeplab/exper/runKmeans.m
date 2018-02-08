function [ segmentation, energies ] = runKmeans( image, initialization)
%RUNKMEANS Summary of this function goes here
%   Detailed explanation goes here
[H W C] = size(image);
segmentation = initialization;
channels = size(initialization, 3);
current_energy = computeKMeans( image, segmentation );
X = zeros(H*W, C);
for c=1:C
    X(:, c) = reshape(image(:,:,c), [H*W 1]);
end

%disp(['initial_energy is ' num2str(current_energy)]);
maxiter = 1000;
energies = current_energy;
for i=1:maxiter
    distance_to_means = zeros(W*H, channels);
    for c=1:channels
        % update mean
        segmentation_c = segmentation(:,:,c);
        segmentation_c = segmentation_c(:);
        segmentation_size = sum(segmentation_c);
        color_sum = X .* repmat(segmentation_c, 1, C);
        color_sum = sum(color_sum);
        mean_color = color_sum / (segmentation_size + realmin);
        distance_to_means(:,c) = sum((X - repmat(mean_color, H*W, 1)).^2,2);
    end
    newsegmentation = segmentation;
    
    newsegmentation(:,:,1) = reshape(double(distance_to_means(:,1) < distance_to_means(:,2)), [H W]);
    newsegmentation(:,:,2) = 1.0 - newsegmentation(:,:,1);
    
    new_energy = computeKMeans( image, newsegmentation );
    if new_energy < current_energy  - 1e-10
        segmentation = newsegmentation;
        current_energy = new_energy;
        energies = [energies current_energy];
        %disp(['current_energy is ' num2str(current_energy)]);
    else
        %disp('converged');
        break;
    end
end
end

