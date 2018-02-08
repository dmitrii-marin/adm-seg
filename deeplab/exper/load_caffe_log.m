function [ logdata ] = load_caffe_log( exp, logfilename )
%LOAD_CAFFE_LOG Summary of this function goes here
%   Detailed explanation goes here
logdir=[exp '/log/deeplab_vgg16/'];
%logfilename='caffe.bin.DRZ-HAL.mtang73.log.INFO.20170923-150454.28599';
% parse file
system(['../code/tools/extra/parse_log.sh ' logdir logfilename  ...
    ' ./']);
%#Iters Seconds TrainingLoss LearningRate
% remove first and last lines
system(['sed ''$d'' ' logfilename '.train > ' logfilename '.train2']);
system(['sed ''1d'' ' logfilename '.train2 > ' logfilename '.train']);
fileID = fopen([logfilename '.train']);
logdata = textscan(fileID,'%f%f%f%f');
fclose(fileID);
% remove data file
system(['rm ' logfilename '.*']);

end

