clc; clear; close all;
input = 'D:\Desktop\DB\samsung\NB\clean\test\up_de\';
% input2 = 'D:\Desktop\DB\Samsung_DB_new\NB\train\decoded\upsampling\';
output = 'C:\Users\NKJ\Desktop\';
% load norm_f_noisy;
% load norm_t_noisy;
% load weights_053_noisy;
% load norm_f_noisy_6300;
% load norm_t_noisy_6300;
% load weights_084_noisy_6300;
% load norm_f_all;
% load norm_t_all;
% load weights_074_all;
load norm_f_1128;
load norm_t_1128;
load weights_094_1128;

Window_w = zeros(512, 1);
ch_tbl = [0 1; 2 3; 4 5; 6 7; 8 9; 10 11; 12 13; 14 16; 17 19; 20 22; 23 26; 27 30; 31 34; 35 39; 40 45;...
    46 52; 53 61; 62 72; 73 87; 88 104; 105 125;];
ch_tbl = ch_tbl+1;
vm_tbl = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7,...
    8, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 24,...
    24, 25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34,35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44, 45,...
    46, 47, 48, 49, 50, 50, 50, 50, 50, 50, 50, 50,50, 50];
LO_CHAN = 1;
HI_CHAN = 21;
Min_Gain = 0.0625;
case1 = 0;
case2 = 0;

%% File
for n = [202]
% for n = 1291
% for n = [6072, 6067, 5262, 5252, 2554, 2544, 2374, 2339]
% for n=861
    a = [input sprintf('up_de_25dB_SS_test_%04d.raw', n)];
%     a = [input2 sprintf('de_25dB_SS_train_%04d.raw', n)];
    b = [output sprintf('H-BWE_%d.raw', n)];
% for n = [1358, 1323, 1159, 1124, 402, 397, 208, 188]
% for n = [3046, 3106, 861, 886]
%     a = [input sprintf('up_de_babble_25dB_SS_test_%04d.raw', n)];
%     a = [input2 sprintf('de_babble_25dB_SS_train_%04d.raw', n)];
%     b = [output sprintf('H-BWE_1_babble_%d.raw', n)];
    fid1 = fopen(a, 'rb');
    signal1 = fread(fid1, inf, 'short');
    fclose(fid1);
    
    alpha = 0.9;
    prev_est_wbmag = zeros(1, 28);
    silence = 8;
    
    fs2 = 16000; framelen2 = 320; overlap2 = 80; nfft2 = 512;
    Deno2 = 0.5 / overlap2;
    for k2 = 1 : overlap2
        f_tmp2 = sin(pi * ((k2 - 1) + 0.5) * Deno2);
        Window_w(k2) = f_tmp2 * f_tmp2;
    end
    for k2 = (overlap2 + 1) : framelen2
        Window_w(k2) = 1.0;
    end
    for k2 = (framelen2 + 1) : (framelen2 + overlap2)
        f_tmp2 = sin(pi * (((k2 - 1) - framelen2 + overlap2) + 0.5) * Deno2);
        Window_w(k2) = f_tmp2 * f_tmp2;
    end
    for k2 = (framelen2 + overlap2 + 1) : nfft2
        Window_w(k2) = 0;
    end
    
    Framenum = floor(length(signal1) / framelen2);
    
    vm = zeros(Framenum, 1);
    VAD = zeros(Framenum, 1);
    frame_energy = zeros(Framenum, 1);
    noisy = zeros(Framenum, 1);
    Sph_Out = zeros(framelen2, 1);
    Sph_Prev = zeros(nfft2, 1);
    Sph_FE_Out = zeros(length(signal1), 1);
    ch_enrg = zeros(21, 1);
    first = 1;
    
    %% Bandwidth extension routine
    for i = 1 : Framenum
        levelpara = zeros(1, 28);
        Sph_Frm1 = zeros(512, 1);
        if(i >= 2)
            Sph_Frm1(1 : overlap2) = signal1((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
        end
        Sph_Frm1(overlap2 + 1 : overlap2 + framelen2) = signal1((i - 1) * framelen2 + 1 : i * framelen2);
        Sph_Frm1 = Sph_Frm1 .* Window_w;
        
        nbfft = fft(Sph_Frm1, 512);
        
        if first == 1
            alpha_bk = 1;
        else
            alpha_bk = 0.55;
        end
        
        for bk = 1 : 21
            enrg = 0;
            j1 = ch_tbl(bk, 1); j2 = ch_tbl(bk, 2);
            
            for jj = j1 : j2
                enrg = enrg + nbfft(jj) .* conj(nbfft(jj));
            end
            enrg = enrg / (j2 - j1 + 1);
            enrg = enrg / (nfft2 / 2);
            ch_enrg(bk) = (1 - alpha_bk) * ch_enrg(bk) + alpha_bk * enrg;
            if(ch_enrg(bk) < Min_Gain)
                ch_enrg(bk) = Min_Gain;
            end
        end
        
        if i <= 4
            for bk = 1 : 21
                ch_noise(bk) = max(ch_enrg(bk), 16);
            end
        end
        
        for bk = LO_CHAN : HI_CHAN
            snr = 10 * log10((ch_enrg(bk) / ch_noise(bk)));
            if (snr < 0)
                snr = 0;
            end
            ch_snr(bk) = (snr + 0.1875) / 0.375;
            ch_snr(bk) = round(ch_snr(bk));
        end
        vm_sum = 0;
        for bk = LO_CHAN : HI_CHAN
            jjj = min(ch_snr(bk), 89);
            vm_sum = vm_sum + vm_tbl(jjj);
        end
        
        vm(i) = vm_sum;
        
        first = 0;
        
        nbfeature = abs(nbfft(1:257, 1)) .^ 2;
        
        if sum(xor(nbfeature(2 : 128, 1), zeros(127, 1))) == 0;
            Sph_FFT = zeros(512, 1);
            prev_est_wbmag = zeros(1, 28);
        else
            nbfeatures = log(nbfeature(2 : 128, 1)');
            
            nbphase = angle(nbfft);
            
            % log power magnitude extension using DNN
            buf1 = bsxfun(@minus, nbfeatures, mean_f);
            data = bsxfun(@rdivide, buf1, std_f);
            data = [data ones(size(data, 1), 1)];
            w1probs = 1 ./ (1 + exp(-data * w1)); w1probs = [w1probs ones(size(w1probs, 1), 1)];
            w2probs = 1 ./ (1 + exp(-w1probs * w2)); w2probs = [w2probs ones(size(w2probs, 1), 1)];
            w3probs = 1 ./ (1 + exp(-w2probs * w3)); w3probs = [w3probs ones(size(w3probs, 1), 1)];
            est_l1 = w3probs * w4;
            buf2 = bsxfun(@times, est_l1, std_t);
            est_l1 = bsxfun(@plus, buf2, mean_t);
            
            est_wbmag_l = [log(nbfeature(1 : 129)') log(nbfeature(128 : -1 : 13)') log(nbfeature(246 : 257)')];
            
            data_band = est_l1;
            
            for k = 1 : 28
                if i == 1
                    prev_est_wbmag = data_band;
                else
                    if data_band(1, k) > prev_est_wbmag(1, k)
                        data_band_1 = prev_est_wbmag(1, k)* (1 - alpha) + alpha * data_band(1, k);
                        levelpara(1, k) = data_band_1 - data_band(1, k);
                        prev_est_wbmag(1, k) = data_band(1, k);
                    else
                        prev_est_wbmag(1, k) = data_band(1, k);
                    end
                end
            end
            
            est_l1 = est_l1 + levelpara;
            
            for k = 1 : 28
                band_energy = mean(est_wbmag_l(134 + 4 * (k - 1) : 137 + 4 * (k - 1))) - est_l1(1, k);
                est_wbmag_l(134 + 4 * (k - 1) : 137 + 4 * (k - 1)) = est_wbmag_l(134 + 4 * (k - 1) : 137 + 4 * (k - 1)) - band_energy;
            end
            
            for k = 1 : 27
                for fb = 1 : 4
                    thres = (est_l1(1, k+1) - est_l1(1, k)) / 4 * (fb - 1) + est_l1(1, k);
                    if thres >= 0
                        if est_wbmag_l(133 + fb + 4 * (k - 1)) > 1.05 * thres
                            est_wbmag_l(133 + fb + 4 * (k - 1)) = 1.05 * thres;
                        end
                    else
                        if est_wbmag_l(133 + fb + 4 * (k - 1)) > 0.95 * thres
                            est_wbmag_l(133 + fb + 4 * (k - 1)) = 0.95 * thres;
                        end
                    end
                end
            end
            
            for fb = 1 : 4
                thres = (est_l1(1, 28) - est_l1(1, 27)) / 4 * (fb + 3) + est_l1(1, 27);
                if thres >= 0
                    if est_wbmag_l(241 + fb) > 1.05 * thres
                        est_wbmag_l(241 + fb) = 1.05 * thres;
                    end
                else
                    if est_wbmag_l(241 + fb) > 0.95 * thres
                        est_wbmag_l(241 + fb) = 0.95 * thres;
                    end
                end
            end
            
            est_wbmag_l = exp(0.5 * est_wbmag_l);
            
            if i > 8
                if vm(i) < 513
                    silence = silence + 1;
                    if silence >= 8
                        VAD(i) = 0;
                    else
                        VAD(i) = 1;
                    end
                else
                    silence = 0;
                    VAD(i) = 1;
                end
            end
            
            frame_energy(i) = sum(est_wbmag_l);
            
            if VAD(i) == 0
                if frame_energy(i) <= 4.896e+04
                    case1 = case1 + 1;
                    case2 = 0;
                    if case1 >= 4
                        noisy(i) = 0;
                    end
                    if case1 == 1
                        est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 1;
                        est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.5;
                    elseif case1 == 2
                        est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.5;
                        est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.25;
                    elseif case1 == 3
                        est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.25;
                        est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.125;
                    elseif case1 >= 4
                        est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.125;
                        est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.0625;
                    end
                elseif frame_energy(i) > 4.896e+04
                    case1 = 0;
                    case2 = case2 + 1;
                    if case2 >= 4
                        noisy(i) = 1;
                    end
                    est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 1;
                    est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.125;
                end
            else
                noisy(i) = noisy(i-1);
                est_wbmag_l = 20 * log10(est_wbmag_l);
                
                buf = est_wbmag_l;
                for fb = 130 : 244
                    if buf(fb - 1) < buf(fb) - 6
                        if buf(fb - 1) == est_wbmag_l(fb - 1)
                            est_wbmag_l(fb - 1) = est_wbmag_l(fb - 1) - 12;
                        end
                    end
                    if buf(fb + 1)<buf(fb) - 6
                        if buf(fb + 1) == est_wbmag_l(fb + 1)
                            est_wbmag_l(fb + 1) = est_wbmag_l(fb + 1) - 12;
                        end
                    end
                end
                
                for fb = 2 : 4
                    his_thres(fb) = 58;
                end
                for fb = 5 : 10
                    his_thres(fb) = 58 - (fb - 4);
                end
                for fb = 11 : 245
                    his_thres(fb) = 52 - 0.0255 * (fb - 10);
                end
                
                for fb = 2 : 128
                    if est_wbmag_l(fb) < his_thres(fb)
                        est_wbmag_l(fb) = est_wbmag_l(fb) - 6;
                    end
                end
                for fb = 129 : 245
                    if est_wbmag_l(fb) < his_thres(fb)
                        est_wbmag_l(fb) = est_wbmag_l(fb) - 12;
                    end
                end
                
                est_wbmag_l = 10 .^ ((est_wbmag_l) / 20);
                
                if noisy(i) == 1
                    est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 1;
                    est_wbmag_l(129 : 245) = est_wbmag_l(129 : 245) * 0.5;
                end
            end
            
            est_wbmag_l(130 : 133) = 0;
            est_wbmag_l(246 : 256) = 0;
            
            est_wbmag_r = fliplr(est_wbmag_l(1, 2 : 256));
            est_wbmag = [est_wbmag_l est_wbmag_r];
            
            wbphase_l = [nbphase(1 : 129, 1)' -nbphase(128 : -1 : 1, 1)'];
            wbphase_r = fliplr(wbphase_l(1, 2 : 256));
            wbphase = [wbphase_l -wbphase_r];
            
            Sph_FFT = est_wbmag' .* exp(1i * wbphase');
        end  % XOR
        
        % Signal synthesis
        Sph_IFFT = real(ifft(Sph_FFT, nfft2));
        Sph_Out(1 : nfft2 - framelen2) = Sph_IFFT(1 : nfft2 - framelen2) + Sph_Prev(1 + framelen2 : nfft2);
        Sph_Out(nfft2 - framelen2 + 1 : framelen2) = Sph_IFFT(nfft2 - framelen2 + 1 : framelen2);
        Sph_Prev = Sph_IFFT;
        Sph_FE_Out((i - 1) * framelen2 + 1 : i * framelen2) = Sph_Out(1 : framelen2);
        
    end  % FRAME
    
    fid2 = fopen(b, 'wb');
    fwrite(fid2, Sph_FE_Out, 'short');
    fclose(fid2);
    
end  % FILE


fclose('all');
disp('FINISH');