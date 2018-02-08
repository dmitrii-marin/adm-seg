clear all
close all
[ logdata ] = load_caffe_log('voc12', 'trainnc3.log' );
% plot
%figure,plot(logdata{1},logdata{2});
%xlabel('Iterations');ylabel('Time (s)');
figure,plot(logdata{1},logdata{3});
xlabel('Iterations');ylabel('Loss');grid on
figure,plot(logdata{1},logdata{3});
xlabel('Iterations');ylabel('Loss');
xlim([100, max(logdata{1})]);grid on
figure,plot(logdata{1},logdata{4});
xlabel('Iterations');ylabel('Learning rate');

[ logdata2 ] = load_caffe_log('voc12',  'trainnc4.log' );
%[ logdata3 ] = load_caffe_log('voc12',  'trainnc5.log' );
figure,plot([logdata{1}; logdata2{1}+20000],...
    [logdata{3}; logdata2{3}]);
xlabel('Iterations');ylabel('Loss');
xlim([100, max(logdata{1})+ max(logdata2{1})]);grid on

[ logdata4 ] = load_caffe_log('voc12', 'trainnc6.log' );
% plot
%figure,plot(logdata{1},logdata{2});
%xlabel('Iterations');ylabel('Time (s)');
figure,plot(logdata4{1},logdata4{3});
xlabel('Iterations');ylabel('Loss');grid on

