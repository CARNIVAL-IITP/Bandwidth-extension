clc; clear; close all

standard_aud=sweeptone(6,1,44100);

[masked_aud, fs] = audioread('./temp/current/면마스크/Audio-02.wav');
% masked_aud = masked_aud(:, 2); 
% plot(masked_aud)
% return;
figure()
hold on 

for i = 2
    [no_masked_aud, fs] = audioread(sprintf('./temp/current/unmasked/Audio-0%d.wav', i));
%     no_masked_aud = no_masked_aud(:, 2); 
    no_mask_rir=impzest(standard_aud, no_masked_aud);
    mask_rir=impzest(standard_aud, masked_aud);
    [value1, n0argmax]=max(no_mask_rir);
    [value2, yesargmax]=max(mask_rir);
    diff=n0argmax-yesargmax;
    if diff<0
        diff = -diff;
        plot(no_mask_rir);
        plot(mask_rir(diff:end));
    else
        plot(no_mask_rir(diff:end));
        plot(mask_rir);
    end
end
legend
legend('RIR', 'Mask + RIR');

result = ifft(fft(no_mask_rir)./fft(mask_rir));

figure()
plot(result)
title('Mask Impulse Response');

freq_imp=fft(mask_rir);
figure()
plot(abs(freq_imp))

freq_imp=fft(no_mask_rir);
figure()
plot(abs(freq_imp))

ttp=fft(mask_rir)./fft(no_mask_rir);
figure()
plot(abs(ttp))

