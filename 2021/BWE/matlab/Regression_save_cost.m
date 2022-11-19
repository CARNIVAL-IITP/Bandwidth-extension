%% Test_DNN_regression & save cost
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016

fclose('all'); close all; clear; clc;

%% Load input and output 
%input_location_feature = ''; %trainset1
input_location_feature = ''; %trainset2
%input_location_target = ''; %testset
%output_location_feature = '';
output_location_feature = '';
%output_location_target = '';

extension = '.mat';


for i=1:40
	% load parameters
    load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/normal_norm_f;
    
    w1=load('PY/FFT64/REGRESSION1/DATASET2/89_w1.txt');
    w2=load('PY/FFT64/REGRESSION1/DATASET2/89_w2.txt');
    w3=load('PY/FFT64/REGRESSION1/DATASET2/89_w3.txt');
    w4=load('PY/FFT64/REGRESSION1/DATASET2/89_w4.txt');
    b1=load('PY/FFT64/REGRESSION1/DATASET2/89_b1.txt');
    b2=load('PY/FFT64/REGRESSION1/DATASET2/89_b2.txt');
    b3=load('PY/FFT64/REGRESSION1/DATASET2/89_b3.txt');
    b4=load('PY/FFT64/REGRESSION1/DATASET2/89_b4.txt');
    
    %Normalized feature(feature extract) x75
    %load([input_location_feature '/' sprintf('train1_feature_%d', i)], sprintf('input1_%d', i));
    load([input_location_feature '/' sprintf('train2_feature1_%d', i)], sprintf('input1_%d', i))
    %load([input_location_normalized_target '/' sprintf('test_feature_%d', i)], sprintf('input1_%d', i));
    input_load=['input = input1_', num2str(i), ';'];
    eval(input_load);
    input_clear=['clear input1_', num2str(i), ';'];
    eval(input_clear);
    len=length(input);
    nbfeatures=input(1:len,:);
    clear input
    
	% Data feedfoward by trained DNN	
    normal_mean_f=repmat(normal_mean_f,len,1);
    buf1 = bsxfun(@minus, nbfeatures, normal_mean_f);
    clear nbfeatures mean_f
    normal_std_f=repmat(normal_std_f,len,1);
    data = bsxfun(@rdivide, buf1, normal_std_f);
    clear buf1 std_f
    b1=repmat(b1,1,len);
    w1probs = max((data*w1+b1'),0);
    clear data w1 b1
    b2=repmat(b2,1,len);
    w2probs = max((w1probs*w2+b2'),0);
    clear w1probs w2 b2
    b3=repmat(b3,1,len);
    w3probs = max((w2probs*w3+b3'),0);
    clear w2probs w3 b3
    b4=repmat(b4,1,len);
    est_l1 = w3probs*w4 + b4';
    clear w3probs w4 b4
    
    %Previous 4 frame log power spectra 16*4 = 64
    input3 = [est_l1(1:end-4,:) est_l1(1:end-3,:) est_l1(1:end-2,:) est_l1(1:end-1,:)];
    %input3 = [est_l1(1:end-4,:) est_l1(2:end-3,:) est_l1(3:end-2,:) est_l1(4:end-1,:)];
    input3_all = ['input3_', num2str(i) ' = input3;'];
    eval(input3_all);
    
	%Save estimated values
    %save([output_location_feature '/' sprintf('train1_inputwbest_%d.mat', i)], sprintf('input3_%d', i));
    save([output_location_feature '/' sprintf('train2_inputwbest_%d.mat', i)], sprintf('input3_%d', i));
    %save([output_location_target '/' sprintf('test_inputwbest_%d.mat', i)], sprintf('input3_%d', i));
    
    %Input = estl1 regression output
    input_load=['input = est_l1', ';'];
    eval(input_load);
    input_clear=['clear est_l1', ';'];
    eval(input_clear);
    
	% Load correct values (WB features) target
    %load([input_location_feature '/' sprintf('train1_feature_%d', i)], sprintf('target_%d', i));
    load([input_location_feature '/' sprintf('train2_feature1_%d', i)], sprintf('target_%d', i));
    %load([input_location_normalized_target '/' sprintf('test_feature_%d', i)], sprintf('target_%d', i));
    target_load=['target = target_', num2str(i), ';'];
    eval(target_load);
    target_clear=['clear target_', num2str(i), ';'];
    eval(target_clear);
    
	% Calculate cost & save
    lambda_1=sqrt(2);
    lambda_2=1;    
    penalty=lambda_1*single(target<input);
    no_penalty=lambda_2*single(target>=input);
    error=input-target;
    cost_save=['cost_', num2str(i), '(:,1) = mean((penalty.*error).^2-(no_penalty.*error).^2, 2);'];
    eval(cost_save);
    
    %save([output_location_feature '/' sprintf('train1_cost_%d.mat', i)], sprintf('cost_%d', i));
    save([output_location_feature '/' sprintf('train2_cost_%d.mat', i)], sprintf('cost_%d', i));
    %save([output_location_target '/' sprintf('test_cost_%d.mat', i)], sprintf('cost_%d', i));
end

clear; clc;
