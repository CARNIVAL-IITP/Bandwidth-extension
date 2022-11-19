%% Extract over-estimation frame
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016

fclose('all'); close all; clear; clc;

%% Load input and output 
input_location_cost = '';
input_location_75_55 = '';
input_location_64 = '';
output_location = '';

% input_location_cost = '';
% input_location_75_55 = '';
% input_location_64 = '';
% output_location = '';

extension = '.mat';

%% Figure histogram (cost) and define over-estimation frame (0.5 % values)
for i = 1:18
%     load([input_location_cost '/' sprintf('train2_cost_%d', i)], sprintf('cost_%d', i));
    load([input_location_cost '/' sprintf('test_cost_%d', i)], sprintf('cost_%d', i));
end

cost = [cost_1; cost_2; cost_3; cost_4; cost_5; cost_6; cost_7; cost_8; cost_9; cost_10; 
        cost_11; cost_12; cost_13; cost_14; cost_15; cost_16; cost_17; cost_18];
    
% cost = [cost_1; cost_2; cost_3; cost_4; cost_5; cost_6; cost_7; cost_8; cost_9; cost_10; 
%        cost_11; cost_12; cost_13; cost_14; cost_15; cost_16; cost_17; cost_18; cost_19;
%        cost_20; cost_21; cost_22; cost_23; cost_24; cost_25; cost_26; cost_27; cost_28; cost_29;
%        cost_30; cost_31; cost_32; cost_33; cost_34; cost_35; cost_36; cost_37; cost_38; cost_39; cost_40];

clear cost_1 cost_2 cost_3 cost_4 cost_5 cost_6 cost_7 cost_8 cost_9 cost_10 cost_11 cost_12 cost_13 cost_14 cost_15 cost_16 cost_17 cost_18

figure()
y = histogram(cost);
title('Histogram of over-estimated measure values')
xlabel('cost')
ylabel('frames')

pdf = y.Values / sum(y.Values);
cdf = zeros(size(pdf));
cdf(1) = pdf(1);

for i = 2:length(pdf)
     cdf(i) = cdf(i-1) + pdf(i);
end

%pdf value
x = 1:1149; %train2
% x = 1:1525; %test2 

figure()
plot((x-1118)*0.099998474, pdf)
figure()
plot((x-1118)*0.099998474, cdf)

%% Save over-estimation frame feautrues & targets (classifcation DB, regression DB)
for i = 1:18
    % Load trainset cost
%     load([input_location_cost '/' sprintf('train2_cost_%d', i)], sprintf('cost_%d', i));
    load([input_location_cost '/' sprintf('test_cost_%d', i)], sprintf('cost_%d', i));
 
    cost_load = ['cost = cost_', num2str(i), ';'];
    eval(cost_load);
    cost = cost(5:end,:);
    cost_clear = ['clear cost_', num2str(i), ';'];
    eval(cost_clear);
    
    % Load trainset input features
%     load([input_location_75_55 '/' sprintf('train2_feature1_%d', i)])
    load([input_location_75_55 '/' sprintf('test_feature_%d', i)])
    input_load = ['input1 = input1_', num2str(i), ';'];
    eval(input_load);
    
    input1 = input1(5:end,:);
    
    input_clear=['clear input1_', num2str(i), ';'];
    eval(input_clear);
    
    input_load = ['input2 = input2_', num2str(i), ';'];
    eval(input_load);
    
    input2 = input2(5:end,:);
    
    input_clear = ['clear input2_', num2str(i), ';'];
    eval(input_clear);
    
%     load([input_location_64 '/' sprintf('train2_inputwbest_%d', i)])
    load([input_location_64 '/' sprintf('test_inputwbest_%d', i)])
    
    input_load = ['input3 = input3_', num2str(i), ';'];
    eval(input_load);
    input_clear = ['clear input3_', num2str(i), ';'];
    eval(input_clear);
    
    %%%%%%%%%% 
%     input = [input1 input2 input3]; %classification DB X194
    input = input1; %regression DB X75
    clear input1 input2 input3
    
%     load([input_location_75_55 '/' sprintf('train2_feature1_%d', i)]);
    load([input_location_75_55 '/' sprintf('test_feature_%d', i)]);
    target_load = ['target = target_', num2str(i), ';'];
    eval(target_load);
    
    target = target(5:end,:);
    
    target_clear = ['clear target_', num2str(i), ';'];
    eval(target_clear);
%     frame_index_normal = single(cost<19.1);
    frame_index_normal = single(cost<25);
    frame_index_c2 = single(find(frame_index_normal(:,1)));
    totnum=size(frame_index_c2,1);
    
    %%%%%%%%%%
    frame_index_c2_new = frame_index_c2(1:ceil((length(frame_index_c2))/(length(input)-length(frame_index_c2))):end,:);
%     frame_index_over = single(cost>=19.1);
    frame_index_over = single(cost>=25);
    frame_index_c3 = single(find(frame_index_over(:,1)));
    [row_c2, col_c2] = size(frame_index_c2_new);
    [row_c3, col_c3] = size(frame_index_c3);
    
    input_new = [input(frame_index_c2_new,:); input(frame_index_c3,:)];
    
    [num_row, num_col] = size(input_new);
    
    %%%%%%%%%% if you make classification DB
%     target_new = zeros(num_row,2); % X2
%     target_new(1:length(frame_index_c2_new),1) = 1; 
%     target_new(length(frame_index_c2_new)+1:length(frame_index_c2_new)+length(frame_index_c3),2) = 1;
    %target_new = [target_new_1, target_new_2];
    
    %%% regression DB
    target_new = [target(frame_index_c2_new,:); target(frame_index_c3,:)];
    input_save = ['input_', num2str(i), '= input_new;'];
    eval(input_save); 
    target_save = ['target_', num2str(i), '= target_new;'];
    eval(target_save);
   
end

input_1 = vertcat( ...
    input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, ...
    input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18 ...
    );
%     input_19, input_20, ...
%     input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, ...
%     input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40 ...
%     );

target_1 = vertcat( ...
    target_1,target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9, target_10, ...
    target_11, target_12, target_13, target_14, target_15, target_16, target_17, target_18 ...
    );
%     target_19,  target_20, ...
%     target_21,  target_22,  target_23,  target_24,  target_25, target_26,  target_27,  target_28,  target_29,  target_30, ...
%     target_31,  target_32,  target_33,  target_34,  target_35,  target_36,  target_37,  target_38,  target_39,  target_40 ...
%     );

% output = fullfile(output_location, ['train2_2classification1' extension]);
% save(output, 'input_1', 'target_1');

% output = fullfile(output_location, ['test_2classification1' extension]);
% save(output, 'input_1', 'target_1');

% output = fullfile(output_location, ['train2_3regression1' extension]);
% save(output, 'input_1', 'target_1');

output = fullfile(output_location, ['test_3regression1' extension]);
save(output, 'input_1', 'target_1');




  


