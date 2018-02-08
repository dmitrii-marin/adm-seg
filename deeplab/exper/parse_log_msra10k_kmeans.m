clear all
close all

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainkmeans.log | awk ''{print $9}''';
[status, results] = system(cmd);
results = str2num(results);
figure,plot(results(2:end));grid on;

xlabel('Iterations');ylabel('K-means Loss');set(gca,'FontSize',16);xlim([-30 1000]);

h=gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,['kmeansloss'],'-dpdf');
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