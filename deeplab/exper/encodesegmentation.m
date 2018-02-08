function [ encodedimg ] = encodesegmentation( image, a )
%ENCODESEGMENTATION Summary of this function goes here
%   Detailed explanation goes here
encodedimg = image;
[H W C] = size(encodedimg);
for h=3:(H-2)
    for w = 3:(W-2)
        if a(h,w)~=a(h+1,w) || a(h,w)~=a(h,w+1) ...
                || a(h,w)~=a(h+2,w) || a(h,w)~=a(h,w+2)
            encodedimg(h,w,:) = [255 0 0 ];
        end
    end
end
return
for h=1:H
    for w = 1:W
        if a(h,w)==1
            c = encodedimg(h,w,:);
            c = c(:);
            encodedimg(h,w,:) = uint8([255 0 0 ]'*0.3 + double(c)*0.7);
        end
    end
end

end

