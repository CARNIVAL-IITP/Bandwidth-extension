clc; clear; close all

aud=sweeptone(6,1,44100);
%audiowrite('sweep.wav',aud, 44100);
figure()
hold on 
legend
subplot(2,1,1)

for i = 2
    [temp_aud, fs] = audioread(sprintf('./temp/current/KF99/Audio-0%i.wav', i));
%     temp_aud = temp_aud(:, 2);
    
    temp_rir=impzest(aud, temp_aud);
    plot(temp_rir);
    title('KF99');
end

subplot(2,1,2)
for i = 2
    [temp_aud, fs] = audioread(sprintf('./temp/current/면마스크/Audio-0%i.wav', i));
%     temp_aud = temp_aud(:, 2);
    
    temp_rir=impzest(aud, temp_aud);
    plot(temp_rir);
    title('면마스크');
end

% figure()
% plot(aud)
% title('Sweeptone');