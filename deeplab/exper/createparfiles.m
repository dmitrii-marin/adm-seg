spatial_w = 0.1;
fileID = fopen('spatial.par','w');
for n=1:21
    fprintf(fileID,'%.2f ',spatial_w);
end
fclose(fileID);

bilateral_w = 0.1;
fileID = fopen('bilateral.par','w');
for n=1:21
    fprintf(fileID,'%.2f ',bilateral_w);
end
fclose(fileID);