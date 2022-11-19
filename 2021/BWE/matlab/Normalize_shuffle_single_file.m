%% Trainset & testset normalize and shuffle
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016
fclose('all'); close all; clear; clc;

% Trainset
input=[];
target=[];

% Normalize
% load MATLAB/FFT64/EXTRACT/DATASET2/train2_2classification1
load MATLAB/FFT64/EXTRACT/DATASET2/train2_3regression1
%load 15_16/LPF7250_2/train_3regression_1

over_mean_f = mean(input_1);
over_std_f = std(input_1);

% save MATLAB/FFT64/NORMALIZE_SINLE/CLASSIFICATION/class_norm_f_1 over_mean_f over_std_f %classification
save MATLAB/FFT64/NORMALIZE_SINGLE/REGRESSION2/over_norm_f_1 over_mean_f over_std_f %regression2
%save 15_16/LPF7250_2/over_norm_f_1 over_mean_f over_std_f;

buf1 = bsxfun(@minus, input_1, over_mean_f);
input = bsxfun(@rdivide, buf1, over_std_f);

% over_mean_t = mean(target_1);
% over_std_t = std(target_1);

% save MATLAB/NORMALIZE_SINGLE/CLASSIFICATION/class_norm_t_1 over_mean_t over_std_t; %classification
%save MATLAB/NORMALIZE_SINGLE/REGRESSION2/over_norm_t_1 over_mean_t over_std_t;

%regression
%save 15_16/LPF7250_2/over_norm_t_1 over_mean_t over_std_t;

% buf2 = bsxfun(@minus, target_1, over_mean_t);
% target = bsxfun(@rdivide, buf2, over_std_t);

% Shuffle
totnum=size(input,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0);
randomorder=randperm(totnum);

input = input(randomorder(1:end),:);
target = target_1(randomorder(1:end),:);

% save MATLAB/FFT64/NORMALIZE_SINGLE/CLASSIFICATION/train2_2classification2 input target %classification
save MATLAB/FFT64/NORMALIZE_SINGLE/REGRESSION2/train2_3regression2 input target %regression
%save 15_16/LPF7250_2/train_3regression_1 input target

clear input target;

% Testset
input=[];
target=[];

% Normalize
% load MATLAB/FFT64/EXTRACT/TEST/test_2classification1
load MATLAB/FFT64/EXTRACT/TEST/test_3regression1

% load MATLAB/EXTRACT/TEST/test_2classification1_1
%load FEATURE_OUTPUT/REGRESSION2/test_3regression_1
%load 15_16/LPF7250_2/test_3regression_1

buf1 = bsxfun(@minus, input_1, over_mean_f);
input = bsxfun(@rdivide, buf1, over_std_f);

% buf2 = bsxfun(@minus, target_1, over_mean_t);
% target = bsxfun(@rdivide, buf2, over_std_t);

% Shuffle
totnum=size(input,1);
fprintf(1, 'Size of the test dataset= %d \n', totnum);

rand('state',0);
randomorder=randperm(totnum);

input = input(randomorder(1:end),:);
target = target_1(randomorder(1:end),:);

% save MATLAB/FFT64/NORMALIZE_SINGLE/CLASSIFICATION/test_2classification2 input target
save MATLAB/FFT64/NORMALIZE_SINGLE/REGRESSION2/test_3regression2 input target
%save 15_16/LPF7250_2/test_3regression_1 input target

clear input target;

%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
fclose('all'); close all;