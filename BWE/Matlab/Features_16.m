%% Features
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016

%% Features
% F1 : Energies of five sub-bands
% F2 : Centroid of the lowband power spectrum
% F3 : Gradient index
% F4 : Spectral flatness
% F5 : Zero crossing rate
% F6 : Kurtosis
% F7 : Variance

function [ output ] = Features (speech_frame, feature, nfft)
%% F1
F1 = [ mean(log(feature(2:4))) mean(log(feature(5:7))) mean(log(feature(8:10))) mean(log(feature(11:13))) mean(log(feature(14:16))) ];

%% F2
numer=0;
denomin=0;
for i = 2 : nfft/4
    numer = numer + (i-1)*8000/15 * feature(i);
end
denomin = (nfft/4+1) * sum(feature(2 : nfft/4));
F2 = (numer / denomin)^2;

%% F3
numer = 0;
denomin = 0;
for i = 2 : 64
    p(i) = sign(speech_frame(i)-speech_frame(i-1));
    if p(i)==0
        p(i)=1;
    end
    if i >= 3
        dp(i) = abs((p(i)-p(i-1))) / 2;
        numer = numer + dp(i) * abs(speech_frame(i)-speech_frame(i-1));
    end
    denomin = sum((speech_frame(1:64)).^2);
end
F3 = numer / sqrt(denomin);

%% F4
geo_mean = nthroot(feature(3), 13);
for i = 4 : 15
    geo_mean = geo_mean * nthroot(feature(i), 13);
end
arith_mean = 1/13 * sum(feature(3:15));
F4 = log10(geo_mean / arith_mean);

%% F5
F5=0;
s(1) = sign(speech_frame(1));
if s(1)==0
    s(1)=1;
end
for i = 2:length(speech_frame)
    s(i) = sign(speech_frame(i));
    if s(i)==0
        s(i)=1;
    end
    F5 = F5 + abs(s(i)-s(i-1)) / (2*(length(speech_frame)-1));
end

%% F6
F6=0;
numer=0;
denomin=0;
for i = 1:length(speech_frame)
    numer = numer + (speech_frame(i))^4 / length(speech_frame);
    denomin = denomin + (speech_frame(i))^2 / length(speech_frame);
end
F6 = numer / denomin^2;

%% F7
F7=0;
F7 = var(log(feature(2:16)));

output = [ F1 F2 F3 F4 F5 F6 F7 ];