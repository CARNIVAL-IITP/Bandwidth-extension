%fs = 16e3;
fs = 44100

% excitation = sweeptone(10, 5, fs, 'SweepFrequencyRange',[10 22500]);
% excitation = sweeptone(5, 10, fs, 'SweepFrequencyRange',[10 22500]);
% excitation = sweeptone(10, 5, fs, 'SweepFrequencyRange',[20 8e3]);
% excitation = sweeptone(5, 10, fs, 'SweepFrequencyRange',[20 8e3]);

%audiowrite('sweep[10_5]_44.1khz.wav', excitation, fs);
%audiowrite('ssweep[5_10]_44.1khz.wav', excitation, fs);
%audiowrite('sweep[10_5]_16khz.wav', excitation, fs);
%audiowrite('sweep[5_10]_16khz.wav', excitation, fs);
