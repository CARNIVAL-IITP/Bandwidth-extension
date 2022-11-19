clc; clear; close all

% aud=sweeptone(6,1,44100);
% aud=sweeptone(8,1,44100);
% aud=sweeptone(8,0.5,44100);
% aud=sweeptone(8,.1,44100);
aud=sweeptone(8,.05,44100);

for i = 0:239
% for i = 0:7
    [masked_aud, fs] = audioread(sprintf('./impulse_input/nomask/%d_nomask.wav', i));
    masked_ir = impzest(aud,masked_aud); 
%     filename = sprintf('./impulse_output/8_1_44100/nomask/impulse_%d.wav', i);
%     filename = sprintf('./impulse_output/8_0.5_44100/nomask/impulse_%d.wav', i);
%     filename = sprintf('./impulse_output/8_0.1_44100/nomask/impulse_%d.wav', i);
    filename = sprintf('./impulse_output/8_0.05_44100/nomask/impulse_%d.wav', i);
    audiowrite(filename, masked_ir, 16000);
end

disp('done!')