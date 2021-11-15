fclose('all'); close all; clear; clc;

input_clean = 'BWE_DB/FEATURE/LPF_16K_RAW/TEST/CLEAN/';
output = 'RESULT/';

load MATLAB/NORMALIZE_SINGLE/CLASSIFICATION/class_norm_f_2;
load FEATURE_OUTPUT/REGRESSION1/normal_norm_f
load FEATURE_OUTPUT/REGRESSION1/normal_norm_t;   
load FEATURE_OUTPUT/REGRESSION2/over_norm_f_1; 
load FEATURE_OUTPUT/REGRESSION2/over_norm_t_1;

w1_normal = load('FEATURE_OUTPUT/parameter1/36_w1.txt'); w2_normal = load('FEATURE_OUTPUT/parameter1/36_w2.txt'); w3_normal = load('FEATURE_OUTPUT/parameter1/36_w3.txt'); w4_normal = load('FEATURE_OUTPUT/parameter1/36_w4.txt');
b1_normal = load('FEATURE_OUTPUT/parameter1/36_b1.txt'); b2_normal = load('FEATURE_OUTPUT/parameter1/36_b2.txt'); b3_normal = load('FEATURE_OUTPUT/parameter1/36_b3.txt'); b4_normal = load('FEATURE_OUTPUT/parameter1/36_b4.txt');
w1_over = load('FEATURE_OUTPUT/parameter3/240_w1.txt'); w2_over = load('FEATURE_OUTPUT/parameter3/240_w2.txt'); w3_over = load('FEATURE_OUTPUT/parameter3/240_w3.txt'); w4_over = load('FEATURE_OUTPUT/parameter3/240_w4.txt');
b1_over = load('FEATURE_OUTPUT/parameter3/240_b1.txt'); b2_over = load('FEATURE_OUTPUT/parameter3/240_b2.txt'); b3_over = load('FEATURE_OUTPUT/parameter3/240_b3.txt'); b4_over = load('FEATURE_OUTPUT/parameter3/240_b4.txt');
w1_class = load('PY/CLASSIFICATION/DATASET1/497_w1.txt'); w2_class = load('PY/CLASSIFICATION/DATASET1/497_w2.txt'); w3_class = load('PY/CLASSIFICATION/DATASET1/497_w3.txt'); w4_class = load('PY/CLASSIFICATION/DATASET1/497_w4.txt');
b1_class = load('PY/CLASSIFICATION/DATASET1/497_b1.txt'); b2_class = load('PY/CLASSIFICATION/DATASET1/497_b2.txt'); b3_class = load('PY/CLASSIFICATION/DATASET1/497_b3.txt'); b4_class = load('PY/CLASSIFICATION/DATASET1/497_b4.txt');

ch_tbl = [0 1; 2 3; 4 5; 6 7; 8 9; 10 11; 12 13; 14 16; 17 19; 20 22; 23 26; 27 30; 31 34; 35 39; 40 45;...
    46 52; 53 61; 62 72; 73 87; 88 104; 105 125;];
min_energy = 0.0625;
vm_tbl = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, ...
    10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 24, 24, ...
    25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44, ...
    45, 46, 47, 48, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50];
fs2 = 16000; framelen2 = 320; overlap2 = 80; nfft2 = 512;
Deno = 0.5 / overlap;
for c1 = 1 : overlap
    f_tmp = sin(pi * ((c1 - 1) + 0.5) * Deno);
    window(c1, 1) = f_tmp * f_tmp;
end
for c1 = (overlap + 1) : framelen
    window(c1, 1) = 1.0;
end
for c1 = (framelen + 1) : (framelen + overlap)
    f_tmp = sin(pi * (((c1 - 1) - framelen + overlap) + 0.5) * Deno);
    window(c1, 1) = f_tmp * f_tmp;
end
for c1 = (framelen + overlap + 1) : nfft
    window(c1, 1) = 0;
end

% hiss_thld(2) = 10 ^ (50 / 20);
% hiss_thld(3) = 10 ^ (44 / 20);
% for fb = 4 : 32
%     hiss_thld(fb) = 10 ^ ((44 - 0.2069 * (fb - 3)) / 20);
% end

N_file = 100;

for n = 1:N_file
    disp(n);
%     a = [input_clean sprintf('up_de_25dB_ss_test_%04d.raw', n)];
%     b = [output sprintf('2016_clean_%d.raw', n)];
    a = [input_clean sprintf('LPF_UP_RAW_test--%d.raw', n)];
    b = [output sprintf('bwe_clean_%d.raw', n)];
    fid = fopen(a, 'rb');
    signal = fread(fid, inf, 'short');
    fclose(fid);
    
    Sph_Out = zeros(framelen, 1);
    Sph_Prev = zeros(nfft, 1);
    Sph_FE_Out = zeros(length(signal), 1);
    prev_est_lps1 = zeros(1, 512);
    ch_energy = zeros(1, 21);
    silence = 40;
    voice = 0;
    silence_clean = 20;
    silence_noisy = 20;
    Framenum = floor(length(signal) / framelen);
    vm_result = zeros(Framenum, 1);
    vm_sum = zeros(Framenum, 1);
    VAD = zeros(Framenum, 1);
    energy_result = zeros(Framenum, 1);
    energy_sum = zeros(Framenum, 1);
    nbfeatures = zeros(1, 75);
    nbfeatures_new = zeros(1, 55);
    
    for i = 1 : Framenum
        Sph_Frm = zeros(512, 1);
        if(i > 1)
            Sph_Frm(1 : overlap) = signal((i - 1) * framelen - overlap  + 1 : (i - 1) * framelen);
        end
        Sph_Frm(overlap + 1 : overlap + framelen) = signal((i - 1) * framelen + 1 : i * framelen);
        Sph_Frm = Sph_Frm .* window;
        
        nbfft = fft(Sph_Frm, 512);
        
        if i > 1
            alpha_ch = 0.55;
        else
            alpha_ch = 1;
        end
        
        vm = 0;
        for ch = 1 : 8
            j1 = ch_tbl(ch, 1); j2 = ch_tbl(ch, 2);
            energy = 0;
            for jj = j1 : j2
                energy = energy + abs(nbfft(jj)).^2;
            end
            energy = energy / (j2 - j1 + 1);
            ch_energy(ch) = max((1 - alpha_ch) * ch_energy(ch) + alpha_ch * energy, min_energy);
            if i < 21
                noise_energy(ch) = max(ch_energy(ch), 8);
            end
            snr(ch) = round(10 * log10((ch_energy(ch) / noise_energy(ch)) / 0.375));
            jj = max(min(snr(ch), 90), 1);
            vm = vm + vm_tbl(jj);
        end
        
        vm_result(i) = vm;
        if i > 20
            vm_sum(i) = vm_sum(i-1) - vm_result(i - 20) + vm_result(i);
        elseif i > 1
            vm_sum(i) = vm_sum(i-1) + vm_result(i);
        else
            vm_sum(i) = vm_result(i);
        end
        
        nbfeature = abs(nbfft(1 : 33, 1)).^2;
        
        if min(nbfeature) ~= 0
            
            nbfeature_new = Features(Sph_Frm, nbfeature(1 : 17), nfft);
            
            nbfeatures = [nbfeatures(1, 16 : 75) log(nbfeature(2 : 16, 1).')];
            nbfeatures_new = [nbfeatures_new(1, 12 : 55) nbfeature_new];
            
            nbphase = angle(nbfft);
            
            if i > 8
                feature=[nbfeatures nbfeatures_new prev_est_lps1];
                
                buf1 = bsxfun(@minus, feature, class_mean_f);
                data = bsxfun(@rdivide, buf1, class_std_f);
                w1probs = max((data * w1_class + b1_class.'), 0);
                w2probs = max((w1probs * w2_class + b2_class.'), 0);
                wclass(i, :) = exp(w2probs * w3_class + b3_class.');
                wclass(i, :) = wclass(i, :) ./ repmat(sum(wclass(i,:), 2), 1, size(wclass(i, :), 2));
                
                buf1 = bsxfun(@minus, nbfeatures, normal_mean_f);
                data = bsxfun(@rdivide, buf1, normal_std_f);
                w1probs = max((data * w1_normal + b1_normal.'), 0);
                w2probs = max((w1probs * w2_normal + b2_normal.'), 0);
                est_lps1 = w2probs * w3_normal + b3_normal.';
                
                prev_est_lps1 = [prev_est_lps1(1, 17 : 512) est_lps1];
                
                buf1 = bsxfun(@minus, nbfeatures, over_mean_f);
                data = bsxfun(@rdivide, buf1, over_std_f);
                w1probs = max((data * w1_over + b1_over.'), 0);
                w2probs = max((w1probs * w2_over + b2_over.'), 0);
                est_lps2 = w2probs * w3_over + b3_over.';
                
                est_lps = est_lps1 * wclass(i, 1) + est_lps2 * wclass(i, 2);
            else
                buf1 = bsxfun(@minus, nbfeatures, normal_mean_f);
                data = bsxfun(@rdivide, buf1, normal_std_f);
                w1probs = max((data * w1_normal + b1_normal.'), 0);
                w2probs = max((w1probs * w2_normal + b2_normal.'), 0);
                est_lps1 = w2probs * w3_normal + b3_normal.';
                
                prev_est_lps1 = [prev_est_lps1(1, 17 : 512) est_lps1];
                
                est_lps = est_lps1;
            end
            
            est_wbmag_l = [log(nbfeature(1 : 16).') est_lps log(nbfeature(33).')];
            est_wbmag_l = exp(0.5 * est_wbmag_l);
            
            if i > 20
                if vm_sum(i) > 413
                    voice = voice + 1;
                    if voice > 5
                        VAD(i) = 1;
                        silence = 0;
                    else
                        VAD(i) = 0;
                    end
                else
                    silence = silence + 1;
                    if silence > 10
                        VAD(i) = 0;
                        voice = 0;
                    else
                        VAD(i) = 1;
                    end
                end
            end
            
            energy_result(i) = sum(est_wbmag_l(2:16).^2);
            if i > 20
                energy_sum(i) = energy_sum(i-1) - energy_result(i - 20) + energy_result(i);
            elseif i > 1
                energy_sum(i) = energy_sum(i-1) + energy_result(i);
            else
                energy_sum(i) = energy_result(i);
            end
            
            if VAD(i) == 0
                if energy_sum(i) > 4.2548e+06
                    silence_clean = 0;
                    silence_noisy = silence_noisy + 1;
                    est_wbmag_l(17 : 32) = est_wbmag_l(17 : 32) * exp(-2.7726 / 10 * silence_noisy);
                else
                    silence_clean = silence_clean + 1;
                    silence_noisy = 0;
                    est_wbmag_l(2 : 16) = est_wbmag_l(2 : 16) * 2 * exp(-2.7726 / 20 * silence_clean);
                    est_wbmag_l(17 : 32) = est_wbmag_l(17 : 32) * exp(-2.7726 / 10 * silence_clean);
                end
%             else
%                 for fb = 17 : 32
%                     if est_wbmag_l(fb) < hiss_thld(fb)
%                         est_wbmag_l(fb) = est_wbmag_l(fb) / 10 ^ (12 / 20);
%                     end
%                 end
            end
            
            est_wbmag_r = fliplr(est_wbmag_l(1, 2 : 32));
            est_wbmag = [est_wbmag_l est_wbmag_r];
            
            wbphase_l = [nbphase(1 : 17, 1).' -nbphase(16 : -1 : 1, 1).'];
            wbphase_r = fliplr(wbphase_l(1, 2 : 32));
            wbphase = [wbphase_l -wbphase_r];
            
            Sph_FFT = est_wbmag.' .* exp(1i * wbphase.');
        else
            Sph_FFT = log((abs(nbfft) .^ 2)) .* exp(1i * angle(nbfft));
        end
        
        Sph_IFFT = real(ifft(Sph_FFT, nfft));
        Sph_Out(1 : nfft - framelen) = Sph_IFFT(1 : nfft - framelen) + Sph_Prev(1 + framelen : nfft);
        Sph_Out(nfft - framelen + 1 : framelen) = Sph_IFFT(nfft - framelen + 1 : framelen);
        Sph_Prev = Sph_IFFT;
        Sph_FE_Out((i - 1) * framelen + 1 : i * framelen) = Sph_Out(1 : framelen);
    end
    
    fid2 = fopen(b, 'wb');
    fwrite(fid2, Sph_FE_Out, 'short');
    fclose(fid2);
end

fclose('all');