fclose('all'); close all; clear; clc;

location = './BWE_DB_new/original/test/Clean_test/';
fs = 16e3;

for i = 1:18
    disp(i)
    for n = 1+(i-1)*100:i*100 
        a = [location sprintf('clean_00%04d.wav', n)];
        fid = fopen(a, 'rb');
        signal = fread(fid, inf, 'short');
        fclose(fid);
        %[signal, fs] = audioread(sprintf('clean_00%04d.wav', n));
        LPF_sig = lowpass(signal, 4000, fs); 
        audiowrite(sprintf('./output/8k_16k_wav-%d.wav', n), LPF_sig, fs);
    end
end