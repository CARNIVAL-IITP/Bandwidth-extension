clc; clear; close all
% show_len=11025


% [aud, fs] = audioread('./temp/temp_with_mask/Audio Track-02.wav');
[masked_aud, fs] = audioread('./temp/current/KF-AD_3D/Audio-02.wav');
% masked_aud = aud(:, 2); 
% return
aud=sweeptone(6,1,44100);

% aud = [aud;aud;aud];
%audiowrite('sweep3.wav', [aud;aud;aud], 16000);

figure()
hold on 
% total_aud = 
for i = 2
%     [temp_aud, fs] = audioread(sprintf('./temp/temp_no_mask/Audio Track-0%d.wav', i));
     [temp_aud, fs] = audioread(sprintf('./temp/current/temp1/Audio-0%d.wav', i));
%     temp_rir=impzest(aud, temp_aud);
%     plot(temp_rir);
    
%     temp_aud=resample(temp_aud, 44100, 16000);
%     if i == 2
%         temp_aud = temp_aud(:, 2); 
%         temp_aud=temp_aud(455936:641130);
        no_mask_rir=impzest(aud, temp_aud);
        mask_rir=impzest(aud, masked_aud);
%           temp_rir=impzest(aud, temp_aud, 'WarmupRuns', 0);
%         temp_rir=temp_rir(1:101);
%         total_rir=temp_rir;
        [value, n0argmax]=max(no_mask_rir);
        [value, yesargmax]=max(mask_rir);
        diff=n0argmax-yesargmax;
    if diff<0
        diff = -diff;
        plot(no_mask_rir);
        plot(mask_rir(diff:end));
    else
        plot(no_mask_rir(diff:end));
        plot(mask_rir);
    end
%     else
%         temp_aud=temp_aud(455936:641130);
%         temp_rir=impzest(aud, temp_aud);
%             temp_rir=impzest(aud, temp_aud, 'WarmupRuns', 0);
%         temp_rir=temp_rir(1:101);
%         plot(temp_rir);
%         total_rir = [total_rir, temp_rir];
    end

% end

legend
legend('RIR', 'Mask + RIR');

result = ifft(fft(no_mask_rir)./fft(mask_rir));

figure()
plot(result)
title('Mask Impulse Response');
 



