%% Normalize & Shuffle
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016
close all; clc;

%input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/TRAIN1'; %trainset1
%input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/TEST'; %testset
%output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/';
%output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/TRAIN2'; %trainset1&2
input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/TEST'; %testset
output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/';
output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

file_list_feature = dir([input_location_feature '/*.mat']);
file_list_target = dir([input_location_target '/*.mat']);
extension = '.mat';



% Compute sub-band energies
for i=1:40
    
    %load([input_location_feature '/' sprintf('train1_feature_%d', i)]);
    load([input_location_feature '/' sprintf('train2_feature_%d', i)]);
    
    input1 = [input1(1:end-4,:) input1(2:end-3,:) input1(3:end-2,:) input1(4:end-1,:) input1(5:end,:)];
    input2 = [input2(1:end-4,:) input2(2:end-3,:) input2(3:end-2,:) input2(4:end-1,:) input2(5:end,:)];
    target = target(5:end,:);
    
    input1_all = ['input1_', num2str(i) ' = input1;'];
    eval(input1_all);
    input2_all = ['input2_', num2str(i) ' = input2;'];
    eval(input2_all);
    target_all = ['target_', num2str(i) ' = target;'];
    eval(target_all);
    
    %save([output_location_feature sprintf('train1_feature1_%d', i)], sprintf('input1_%d', i), sprintf('input2_%d', i), sprintf('target_%d', i));
    save([output_location_feature sprintf('train2_feature1_%d', i)], sprintf('input1_%d', i), sprintf('input2_%d', i), sprintf('target_%d', i));
end



for i=1:18
    
    load([input_location_target '/' sprintf('test_feature_%d', i)]);
    
    input1 = [input1(1:end-4,:) input1(2:end-3,:) input1(3:end-2,:) input1(4:end-1,:) input1(5:end,:)];
    input2 = [input2(1:end-4,:) input2(2:end-3,:) input2(3:end-2,:) input2(4:end-1,:) input2(5:end,:)];
    target = target(5:end,:);
    
    input1_all = ['input1_', num2str(i) ' = input1;'];
    eval(input1_all);
    input2_all = ['input2_', num2str(i) ' = input2;'];
    eval(input2_all);
    target_all = ['target_', num2str(i) ' = target;'];
    eval(target_all);
        
    save([output_location_target '/' sprintf('test_feature1_%d', i)], sprintf('input1_%d', i), sprintf('input2_%d', i), sprintf('target_%d', i));
    
end

% Calculate mean & standard deviation parameters

train_len=zeros(10,1);
test_len=zeros(2,1);

for i=1:40
    
    %load([output_location_feature '/' sprintf('train1_feature_%d', i)]);
    load([output_location_feature '/' sprintf('train2_feature1_%d', i)])
    
    mean_f_save=['mean_f_', num2str(i), '=mean(input1_', num2str(i), ');'];
    eval(mean_f_save);
    var_f_save=['var_f_', num2str(i), '=var(input1_', num2str(i), ');'];
    eval(var_f_save);
    
    mean_t_save=['mean_t_', num2str(i), '=mean(target_', num2str(i), ');'];
    eval(mean_t_save);
    var_t_save=['var_t_', num2str(i), '=var(target_', num2str(i), ');'];
    eval(var_t_save);
    
    len_save=['train_len(', num2str(i), ',1)=length(input1_', num2str(i), ');'];
    eval(len_save);
    
    clear(sprintf('input1_%d', i), sprintf('target_%d', i));
    
end

%train_len_name = fullfile(output_location_feature, ['train1_len' extension]);
train_len_name = fullfile(output_location_feature, ['train2_len' extension]);
save(train_len_name, 'train_len');

%normal_mean_f
normal_mean_f=zeros(1, size(mean_f_1,2));

for i=1:40
    mean_f_save=['normal_mean_f=normal_mean_f+train_len(i)*mean_f_', num2str(i), ';'];
    eval(mean_f_save);
end

normal_mean_f=normal_mean_f/sum(train_len);
normal_mean_f = single(normal_mean_f);
var_f=zeros(1, size(var_f_1,2));

for i=1:40
    var_f_save=['var_f=var_f+train_len(i)*var_f_', num2str(i), ';'];
    eval(var_f_save);
end

var_f=var_f/sum(train_len);
normal_std_f = sqrt(var_f);
normal_std_f = single(normal_std_f);

normal_norm_f_name = fullfile(output_location_feature, ['normal_norm_f' extension]);
save(normal_norm_f_name, 'normal_mean_f', 'normal_std_f');

%normal_mean_t

load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/train1_len
normal_mean_t=zeros(1, size(mean_t_1,2));
for i=1:length(40) %feature
 
    mean_t_save=['normal_mean_t=normal_mean_t+train_len(i)*mean_t_', num2str(i), ';'];
    eval(mean_t_save);
end

normal_mean_t=normal_mean_t/sum(train_len);
normal_mean_t = single(normal_mean_t);
var_t=zeros(1, size(var_t_1,2));

for i=1:length(40) %feature
  
    var_t_save=['var_t=var_t+train_len(i)*var_t_', num2str(i), ';'];
    eval(var_t_save);
end

var_t=var_t/sum(train_len);
normal_std_t = sqrt(var_t);
normal_std_t = single(normal_std_t);

normal_norm_t_name = fullfile(output_location_feature, ['normal_norm_t' extension]);
save(normal_norm_t_name, 'normal_mean_t', 'normal_std_t');
%}

% Normalize data by mean & std parameters

clear

%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/train1_len
%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/normal_norm_f
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/train2_len
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/normal_norm_f

%input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN1'; %trainset1
%input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TEST'; %testset
%output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/';
%output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/';
output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';
extension = '.mat';

for i=1:40
    %load([output_location_feature '/' sprintf('train1_feature_%d', i)]);
    load([output_location_feature '/' sprintf('train2_feature1_%d', i)]);
    
    buf = ['buf1 = bsxfun(@minus, input1_', num2str(i), ', normal_mean_f);'];
    eval(buf);
    input = ['input1_', num2str(i), '= bsxfun(@rdivide, buf1, normal_std_f);'];
    eval(input);
    
    %save([output_location_feature sprintf('train1_1regression1_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
    save([output_location_feature sprintf('train2_1regression1_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
    
    clear(sprintf('input1_%d', i), sprintf('target_%d', i));
end

clear

%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/train1_len
%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/normal_norm_f
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/train2_len
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/normal_norm_f

%input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN1'; %trainset1
%input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TEST'; %testset
%output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/';
%output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN2'; %trainset1&2
input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TEST'; %testset
output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/';
output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

extension = '.mat';


for i=1:18
    
    load([output_location_target '/' sprintf('test_feature_%d', i)]);
    
    buf = ['buf1 = bsxfun(@minus, input1_', num2str(i), ', normal_mean_f);'];
    eval(buf);
    input = ['input1_', num2str(i), '= bsxfun(@rdivide, buf1, normal_std_f);'];
    eval(input);
    
    len_save=['test_len(', num2str(i), ',1)=length(input1_', num2str(i), ');'];
    eval(len_save);
    
    save([output_location_target '/' sprintf('test_1regression1_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
    clear(sprintf('input1_%d', i), sprintf('target_%d', i));
end

test_len_name = fullfile(output_location_feature, ['test_len' extension]);
save(test_len_name, 'test_len');

% Shuffle the trainset
clear


%input_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1'; %trainset1
%input_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST'; %testset
%output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/';
%output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

input_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/'; %trainset1
input_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST'; %testset
output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/';
output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

extension = '.mat';

%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/train1_len
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/train2_len

for i=1:40
    disp(i)
    for j=1:40
        if(i~=j)
            %load([input_location_feature '/' sprintf('train1_1regression1_%d', i)]);
            %load([input_location_feature '/' sprintf('train1_1regression1_%d', j)]);
            load([input_location_feature '/' sprintf('train2_1regression1_%d', i)]);
            load([input_location_feature '/' sprintf('train2_1regression1_%d', j)]);
            input_all = ['input1 = [input1_', num2str(i), ';input1_', num2str(j), '];'];
            eval(input_all);
            target_all = ['target = [target_', num2str(i), ';target_', num2str(j), '];'];
            eval(target_all);
            
            totnum=size(input1,1);
            
            rand('state',0); %so we know the permutation of the training data
            randomorder=randperm(totnum);
            
            input1 = input1(randomorder(1:totnum), :);
            target = target(randomorder(1:totnum), :);
            
            input_all = ['input1_', num2str(i), '= input1(1:train_len(i),:);'];
            eval(input_all);
            target_all = ['target_', num2str(i), '= target(1:train_len(i),:);'];
            eval(target_all);
            input_all = ['input1_', num2str(j), '= input1(1+train_len(i):end,:);'];
            eval(input_all);
            target_all = ['target_', num2str(j), '= target(1+train_len(i):end,:);'];
            eval(target_all);
            
            %save([output_location_feature sprintf('train1_1regression2_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
            %save([output_location_feature sprintf('train1_1regression2_%d', j)], sprintf('input1_%d', j), sprintf('target_%d', j));
            save([output_location_feature sprintf('train2_1regression2_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
            save([output_location_feature sprintf('train2_1regression2_%d', j)], sprintf('input1_%d', j), sprintf('target_%d', j));
  
            clear('input1', 'target', sprintf('input1_%d', i), sprintf('target_%d', i), sprintf('input1_%d', j), sprintf('target_%d', j));
            
            %%% Reset random seeds
            rand('state',sum(100*clock));
            randn('state',sum(100*clock));
        end
    end    
end

% Shuffle the testset
clear

input_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1'; %trainset1
input_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST'; %testset
output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/';
output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

%input_location_feature = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN2'; %trainset1&2
%input_location_target = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TEST'; %testset
%output_location_feature = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/';
%output_location_target = 'MATLAB/FFT64/NORMALIZE_MULTIPLE/TEST/';

extension = '.mat';
load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET1/test_len
%load MATLAB/FFT64/NORMALIZE_MULTIPLE/REGRESSION1_DATASET2/test_len

for i=1:18
    disp(i)
    for j=1:18
        if(i~=j)
            
            load([input_location_target '/' sprintf('test_1regression1_%d', i)]);
            load([input_location_target '/' sprintf('test_1regression1_%d', j)]);
            
            input_all = ['input1 = [input1_', num2str(i), ';input1_', num2str(j), '];'];
            eval(input_all);
            target_all = ['target = [target_', num2str(i), ';target_', num2str(j), '];'];
            eval(target_all);
            
            totnum=size(input1,1);
            
            rand('state',0); %so we know the permutation of the training data
            randomorder=randperm(totnum);
            
            input1 = input1(randomorder(1:totnum), :);
            target = target(randomorder(1:totnum), :);
            
            input_all = ['input1_', num2str(i), '= input1(1:test_len(i),:);'];
            eval(input_all);
            target_all = ['target_', num2str(i), '= target(1:test_len(i),:);'];
            eval(target_all);
            input_all = ['input1_', num2str(j), '= input1(1+test_len(i):end,:);'];
            eval(input_all);
            target_all = ['target_', num2str(j), '= target(1+test_len(i):end,:);'];
            eval(target_all);
            
            save([output_location_target '/' sprintf('test_1regression2_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
            save([output_location_target '/' sprintf('test_1regression2_%d', i)], sprintf('input1_%d', i), sprintf('target_%d', i));
            %save(sprintf('FEATURE_OUTPUT/REGRESSION1/test_1regression_%d', i), sprintf('input_%d', i), sprintf('target_%d', i));
            %save(sprintf('FEATURE_OUTPUT/REGRESSION1/test_1regression_%d', j), sprintf('input_%d', j), sprintf('target_%d', j));
            
            clear('input1', 'target', sprintf('input1_%d', i), sprintf('target_%d', i), sprintf('input1_%d', j), sprintf('target_%d', j));
            
            %%% Reset random seeds
            rand('state',sum(100*clock));
            randn('state',sum(100*clock));
        end
    end    
end