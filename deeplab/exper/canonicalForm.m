function [ newblob ] = canonicalForm( blob )
%CANONICALFORM Summary of this function goes here
%   Detailed explanation goes here
% blob is of shape W * H * C
% newblob is of shape H * W * C
[W H C] = size(blob);
newblob = zeros(H, W, C);
for c=1:C
    newblob(:,:,c) = blob(:,:,c)';
end

end

