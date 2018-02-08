clear all
close all

cmd = 'grep '', loss = '' pascal_scribble/log/deeplab_largeFOV/train.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax Loss');title('Baseline');

cmd = 'grep '', loss = '' pascal_scribble/log/deeplab_largeFOV/trainwithncloss.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax + NC Loss');title('Train with NC loss')

cmd = 'grep '': nc_loss = '' pascal_scribble/log/deeplab_largeFOV/trainwithncloss.log | awk ''{print $11}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NC Loss');title('Train with NC loss')

cmd = 'grep '': softmax_loss = '' pascal_scribble/log/deeplab_largeFOV/trainwithncloss.log | awk ''{print $11}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax Loss');title('Train with NC loss')

return

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeansoneimage1e-4.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;%xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Kmeans Loss');title('Train with one image');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeansoneimage1e-5.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;%xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Kmeans Loss');title('Train with one image');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeansoneimage1e-6.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;%xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Kmeans Loss');title('Train with one image');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeansoneimage1e-7.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;%xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Kmeans Loss');title('Train with one image');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeansoneimage1e-8.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;%xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Kmeans Loss');title('Train with one image');

legend('lr=1e-4','lr=1e-5','lr=1e-6','lr=1e-7','lr=1e-8');
