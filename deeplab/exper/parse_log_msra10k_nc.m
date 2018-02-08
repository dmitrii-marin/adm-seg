clear all
close all

cmd = 'grep '', loss = '' msra10k/log/deeplab_largeFOV/trainnc.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss');title('Normalized Cut Only');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainnc.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Normalized Cut Loss');set(gca,'FontSize',16);xlim([-30 1000]);

h=gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,['ncloss'],'-dpdf');


cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainnc_new.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results)+2);grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss');title('Normalized Cut Only');

return


cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainnc1e-4.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss');title('Normalized Cut Only');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainnc5e-4.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss');title('Normalized Cut Only');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainncrgb1e-6.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss (RGB)');title('Normalized Cut Only (RGB)');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainncrgb1e-7.log | awk ''{print $9}''';
[status, results] = system(cmd);
hold on,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss (RGB)');title('Normalized Cut Only (RGB)');

legend('initial lr = 1e-6','initial lr = 1e-7');

cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainncrgb2.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut Loss (RGB)');title('Normalized Cut Only (RGB)');

return
cmd = 'grep ''Train net output #0: accuracy = '' msra10k/log/deeplab_vgg16/train.log | awk ''{print $11}''';
[status, results] = system(cmd);
plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);

cmd = 'grep ''Train net output #0: accuracy = '' msra10k/log/deeplab_vgg16/trainwnc.log | awk ''{print $11}''';
[status, results] = system(cmd);
hold on 
plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);

ylabel('accuracy');
legend('Baseline','With Normalized Cut Loss');
grid on

% baseline
cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/train.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax Loss');title('Baseline');

% train with nc loss
cmd = 'grep '', loss = '' msra10k/log/deeplab_vgg16/trainwnc.log | awk ''{print $9}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax and normalized cut loss');title('With Normalized Cut Loss');

cmd = 'grep ''Train net output #4: softmax_loss = '' msra10k/log/deeplab_vgg16/trainwnc.log | awk ''{print $11}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('Softmax loss');title('With Normalized Cut Loss');

cmd = 'grep ''Train net output #3: nc_loss = '' msra10k/log/deeplab_vgg16/trainwnc.log | awk ''{print $11}''';
[status, results] = system(cmd);
figure,plot(str2num(results));grid on;xlim([0 numel(str2num(results))]);
xlabel('Iterations');ylabel('NormalizedCut loss');title('With Normalized Cut Loss');





%[ log ] = load_caffe_log('msra10k', 'train.log' );
%figure,plot(log{1}, log{3});
%xlabel('Iterations');ylabel('Cross Entropy');

return

[ lognc ] = load_caffe_log('msra10k', 'trainnc.log' );
figure,plot(lognc{1}, lognc{3});
xlabel('Iterations');ylabel('Normalized Cut Loss');
grid on

[ logwnc ] = load_caffe_log('msra10k', 'trainwnc.log' );
figure,plot(logwnc{1}, logwnc{3});
xlabel('Iterations');ylabel('Cross Entropy + Normalized Cut');
grid on

grid on

return

% baseline
[ logdata ] = load_caffe_log('msra10k', 'train.log' );
% plot
%figure,plot(logdata{1},logdata{2});
%xlabel('Iterations');ylabel('Time (s)');
%figure,plot(logdata{1},logdata{3});
%xlabel('Iterations');ylabel('Cross Entropy Loss');grid on
figure,plot(logdata{1},logdata{3});
xlabel('Iterations');ylabel('Cross Entropy Loss');
xlim([100, max(logdata{1})]);grid on
title('baseline');
% baseline with nc loss
[ logdata ] = load_caffe_log('msra10k', 'trainnc.log' );
% plot
%figure,plot(logdata{1},logdata{2});
%xlabel('Iterations');ylabel('Time (s)');
%figure,plot(logdata{1},logdata{3});
%xlabel('Iterations');ylabel('Cross Entropy Loss');grid on
figure,plot(logdata{1},logdata{3});
xlabel('Iterations');ylabel('Cross Entropy + NC');
xlim([100, max(logdata{1})]);grid on
title('cross entropy + NC');

% nc only
[ logdata ] = load_caffe_log('msra10k', 'trainnconly.log' );
% plot
%figure,plot(logdata{1},logdata{2});
%xlabel('Iterations');ylabel('Time (s)');
%figure,plot(logdata{1},logdata{3});
%xlabel('Iterations');ylabel('Cross Entropy Loss');grid on
figure,plot(logdata{1},logdata{3});
xlabel('Iterations');ylabel('NC');
%xlim([100, max(logdata{1})]);grid on
title('NC');
return;
