fclose('all'); close all; clear; clc;


%input_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/TRAIN1';
%input_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/TRAIN2';
input_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/TEST';
%output_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN1';
%output_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TRAIN2';
output_folder_path = 'MATLAB/FFT64/FEATURE_EXTRACT/INPUT1/TEST';

file_list = dir([input_folder_path '/*.mat']);

for i=1:length(file_list)
    %load([input_folder_path '/' sprintf('train1_feature_%d', i)]);
    %load([input_folder_path '/' sprintf('train2_feature_%d', i)]);
    load([input_folder_path '/' sprintf('test_feature_%d', i)]);
    
    input1_rename = ['input_', num2str(i) ' = input1;'];
    eval(input1_rename);
    
    %input2_rename = ['input_', num2str(i) ' = input2;'];
    %eval(input2_rename);
    
    target_rename = ['target_', num2str(i) ' = target;'];
    eval(target_rename);
    
    %save([output_folder_path '/' sprintf('train1_feature_%d', i)], sprintf('input_%d', i),sprintf('target_%d', i));
    %save([output_folder_path '/' sprintf('train2_feature_%d', i)], sprintf('input_%d', i),sprintf('target_%d', i));
    save([output_folder_path '/' sprintf('test_feature_%d', i)], sprintf('input_%d', i),sprintf('target_%d', i));
end

fclose('all'); close all; clear
