clc; clear; close all;
 
win_size = 0.01;
fft_overlap = 0.5;
Fs = 16000;

a = 'RESULT/PLOT/original1.raw';
b ='RESULT/PLOT/lpf.raw';
c ='RESULT/PLOT/output_1.raw';

fid = fopen(a, 'rb');
original_signal = fread(fid, inf, 'short');
fclose(fid);
fid = fopen(b, 'rb');
lpf_signal = fread(fid, inf, 'short');
fclose(fid);
fid = fopen(c, 'rb');
BWE_signal = fread(fid, inf, 'short');
fclose(fid);

% signal = signal(:,1); %use only the left channel 
hop_size = Fs*win_size;
nfft = hop_size/fft_overlap;
noverlap = nfft-hop_size;
w = sqrt(hann(nfft)); %use some window 
%Normal Spectrogram plot
subplot(3,1,1);
spectrogram(original_signal, w ,noverlap, nfft, Fs, 'yaxis' );
colormap jet;
caxis([-10 50])
title('Original signal spectrogram plot');

fid = fopen(a, 'rb');
original_signal = fread(fid, inf, 'short');
fclose(fid);
fid = fopen(b, 'rb');
lpf_signal = fread(fid, inf, 'short');
fclose(fid);
fid = fopen(c, 'rb');
BWE_signal = fread(fid, inf, 'short');
fclose(fid);

% signal = signal(:,1); %use only the left channel 
hop_size = Fs*win_size;
nfft = hop_size/fft_overlap;
noverlap = nfft-hop_size;
w = sqrt(hann(nfft)); %use some window 
%Normal Spectrogram plot
subplot(3,1,2);
spectrogram(original_signal, w ,noverlap, nfft, Fs, 'yaxis' );
colormap jet;
caxis([-30 50])
title('Original signal spectrogram plot');

%{
hop_size = Fs*win_size;
nfft = hop_size/fft_overlap;
noverlap = nfft-hop_size;
w = sqrt(hann(nfft)); %use some window 
%Normal Spectrogram plot
subplot(3,1,2);
spectrogram(lpf_signal, w ,noverlap, nfft, Fs, 'yaxis' );
colormap jet;
title('Low pass filtered signal spectrogram plot');

hop_size = Fs*win_size;
nfft = hop_size/fft_overlap;
noverlap = nfft-hop_size;
w = sqrt(hann(nfft)); %use some window 
%Normal Spectrogram plot
subplot(3,1,3);
spectrogram(BWE_signal, w ,noverlap, nfft, Fs, 'yaxis' );
colormap jet;
title('BWE signal spectrogram plot');
%}