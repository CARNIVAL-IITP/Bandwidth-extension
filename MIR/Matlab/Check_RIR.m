clc; clear; close all

aud=sweeptone(6,1,44100);

[temp_aud, fs] = audioread('./temp/blocked_409/Audio Track-02.wav');
%current/temp18/Audio-02.wav');

figure()
plot(temp_aud);
hold on 

temp_rir=impzest(aud, temp_aud);
figure()
plot(temp_rir);

%%24?
