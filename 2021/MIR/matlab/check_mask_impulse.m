clc; clear; close all

aud=sweeptone(10,5,44100);

[temp_aud, fs] = audioread('./impulse_output/fabric/10_masked_fabric.wav');

figure()
plot(sweeptone);
hold on 

temp_rir=impzest(aud, temp_aud);
figure()
plot(temp_rir);

