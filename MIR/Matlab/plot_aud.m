[aud, fs] = audioread('./temp/temp_with_mask/Audio Track-02_dc.wav');
plot(aud);
%fs = 16e3;
%excitation = sweeptone(10, 5, fs, 'SweepFrequencyRange',[20 8e3]);
%audiowrite('sweep_16khz.wav', excitation, fs);
aud=sweeptone(6,1,44100);
[temp_aud, fs] = audioread(sprintf('/home/utahboy3502/1007MIR/MIR/Impulse_test-01.wav'));
ir = impzest(aud,temp_aud);
plot(ir)
audiowrite('ir_test.wav', ir, fs);
