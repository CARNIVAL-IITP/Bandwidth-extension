%% Data pre-processing tool
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016
% Over-estimation ���� �� Feature Extraction 
% 2016-11-10

fclose('all'); close all; clear; clc;

load FEATURE_OUTPUT/REGRESSION1/normal_norm_f;   % Input feature normalization
load FEATURE_OUTPUT/REGRESSION1/normal_norm_t;   % Target feature normalization
load FEATURE_OUTPUT/REGRESSION2/over_norm_f_1;
load FEATURE_OUTPUT/REGRESSION2/over_norm_t_1;
load MATLAB/NORMALIZE_SINGLE/CLASSIFICATION/class_norm_f_2;
%FEATURE_OUTPUT/CLASSIFICATION/class_norm_f_2;

%{
load FEATURE_OUTPUT/REGRESSION1/normal_norm_f;   % Input feature normalization
load FEATURE_OUTPUT/REGRESSION1/normal_norm_t;   % Target feature normalization
load FEATURE_OUTPUT/REGRESSION2/over_norm_f_1;
load FEATURE_OUTPUT/REGRESSION2/over_norm_t_1;
load FEATURE_OUTPUT/CLASSIFICATION/class_norm_f_2;
%}

w1_normal = load('FEATURE_OUTPUT/parameter1/36_w1.txt'); w2_normal = load('FEATURE_OUTPUT/parameter1/36_w2.txt'); w3_normal = load('FEATURE_OUTPUT/parameter1/36_w3.txt'); w4_normal = load('FEATURE_OUTPUT/parameter1/36_w4.txt');
b1_normal = load('FEATURE_OUTPUT/parameter1/36_b1.txt'); b2_normal = load('FEATURE_OUTPUT/parameter1/36_b2.txt'); b3_normal = load('FEATURE_OUTPUT/parameter1/36_b3.txt'); b4_normal = load('FEATURE_OUTPUT/parameter1/36_b4.txt');
w1_over = load('FEATURE_OUTPUT/parameter3/240_w1.txt'); w2_over = load('FEATURE_OUTPUT/parameter3/240_w2.txt'); w3_over = load('FEATURE_OUTPUT/parameter3/240_w3.txt'); w4_over = load('FEATURE_OUTPUT/parameter3/240_w4.txt');
b1_over = load('FEATURE_OUTPUT/parameter3/240_b1.txt'); b2_over = load('FEATURE_OUTPUT/parameter3/240_b2.txt'); b3_over = load('FEATURE_OUTPUT/parameter3/240_b3.txt'); b4_over = load('FEATURE_OUTPUT/parameter3/240_b4.txt');
w1_class = load('PY/CLASSIFICATION/DATASET1/497_w1.txt'); w2_class = load('PY/CLASSIFICATION/DATASET1/497_w2.txt'); w3_class = load('PY/CLASSIFICATION/DATASET1/497_w3.txt');
b1_class = load('PY/CLASSIFICATION/DATASET1/497_b1.txt'); b2_class = load('PY/CLASSIFICATION/DATASET1/497_b2.txt'); b3_class = load('PY/CLASSIFICATION/DATASET1/497_b3.txt');

%{
w1_normal = load('FEATURE_OUTPUT/36_w1.txt'); w2_normal = load('FEATURE_OUTPUT\36_w2.txt'); w3_normal = load('FEATURE_OUTPUT/36_w3.txt'); w4_normal = load('FEATURE_OUTPUT/36_w4.txt');
b1_normal = load('15_16\226_b1.txt'); b2_normal = load('15_16\226_b2.txt'); b3_normal = load('15_16\226_b3.txt'); b4_normal = load('15_16\226_b4.txt');
w1_over = load('15_16\86_w1.txt'); w2_over = load('15_16\86_w2.txt'); w3_over = load('15_16\86_w3.txt'); w4_over = load('15_16\86_w4.txt');
b1_over = load('15_16\86_b1.txt'); b2_over = load('15_16\86_b2.txt'); b3_over = load('15_16\86_b3.txt'); b4_over = load('15_16\86_b4.txt');
w1_class = load('15_16\130_w1.txt'); w2_class = load('15_16\130_w2.txt'); w3_class = load('15_16\130_w3.txt');
b1_class = load('15_16\130_b1.txt'); b2_class = load('15_16\130_b2.txt'); b3_class = load('15_16\130_b3.txt');
%}

Window_w = zeros(512, 1);
ch_tbl = [5 8; 9 12; 13 16; 17 20; 21 24; 25 28; 29 34; 35 40; 41 46; 47 54; 55 62; 63 72; 73 84; 85 98; 99 112; 113 128];
min_energy = 0.0625;
vm_tbl = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, ...
    10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 24, 24, ...
    25, 26, 27, 28, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44, ...
    45, 46, 47, 48, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50];
case1 = 0; case2 = 0;

cnt_1 = 0;
cnt_2 = 0;
cnt_3 = 0;
cnt_4 = 0;
cnt_5 = 0;

over_nbfeature_temp_total   = [];
over_target_temp_total      = [];

for file = 1   % Training DB 
    if file == 1 || file == 2
        N_file = 1800;
        %6400
    elseif file == 3 || file == 4
        N_file = 962;
    elseif file == 5 || file == 6
        N_file = 7560;
    else
        disp('ERROR');
        pause;
    end
  
    if file == 1   % Samsung Clean DB
        location_feature = 'BWE_DB/FEATURE/LPF_16K_RAW/TEST/CLEAN/';
        location_target_1 = 'BWE_DB/TARGET/ORIGINAL_16K_RAW/TEST/CLEAN/';
        %location_target_2 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6750\';
        %location_target_3 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6500\';
        %location_target_4 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6250\';
        
        %if file == 1   % Samsung Clean DB
        %location_feature = 'H:\2016_�Ｚ����\DB\samsung\NB\clean\train\up_de\';./BWE_DB_new/LPF_UP_RAW/test/clean/
        %location_target_1 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF7000\';
        %location_target_2 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6750\';
        %location_target_3 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6500\';
        %location_target_4 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6250\';
        
    elseif file == 2   % Samsung Babble DB
        location_feature = 'H:\2016_�Ｚ����\DB\samsung\NB\babble\train\up_de\';
        location_target_1 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF7000\';
        location_target_2 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6750\';
        location_target_3 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6500\';
        location_target_4 = 'H:\2016_�Ｚ����\DB\samsung\WB\clean\train\LPF6250\';
    elseif file == 3   % NTT Clean DB
        location_feature = 'H:\2016_�Ｚ����\DB\ntt\NB\clean\train\up_de\';
        location_target_1 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF7000\';
        location_target_2 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6750\';
        location_target_3 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6500\';
        location_target_4 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6250\';
    elseif file == 4   % NTT Babble DB
        location_feature = 'H:\2016_�Ｚ����\DB\ntt\NB\babble\train\up_de\';
        location_target_1 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF7000\';
        location_target_2 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6750\';
        location_target_3 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6500\';
        location_target_4 = 'H:\2016_�Ｚ����\DB\ntt\WB\clean\train\LPF6250\';
    elseif file == 5   % ETRI Clean DB
        location_feature = 'H:\2016_�Ｚ����\DB\etri\NB\clean\train\up_de\';
        location_target_1 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF7000\';
        location_target_2 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF6750\';
        location_target_3 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF6500\';
        location_target_4 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF6250\';
    elseif file == 6   % ETRI Babble DB
        location_feature = 'H:\2016_�Ｚ����\DB\etri\NB\babble\train\up_de\';
        location_target_1 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF7000\';
        location_target_2 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF7500\';
        location_target_3 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF7000\';
        location_target_4 = 'H:\2016_�Ｚ����\DB\etri\WB\clean\train\LPF6500\';
    else
        disp('ERROR');
        pause;
    end
    
    output = 'RESULT/';
    %output = 'H:\2016_�Ｚ����\Results\';

    over_nbfeature_temp_DB      = [];
    over_target_temp_DB  = [];
    %% File
    for n = 1:N_file

        disp(n);
        clear cost_1 cost_2 cost_3 Sph_Out Sph_Prev Sph_FE_Out noisy vm_result VAD frame_energy

        % Load file & delay calculation
       
        if file == 1   % Samsung Clean DB
            a = [location_feature sprintf('LPF_UP_RAW_test--%d.raw', n)];
            b = [location_target_1 sprintf('16k_raw_test-%d.raw', n)];
            %c = [location_target_2 sprintf('25dB_ss_train_%04d.raw', n)];
            %d = [location_target_3 sprintf('25dB_ss_train_%04d.raw', n)];
            %e = [location_target_4 sprintf('25dB_ss_train_%04d.raw', n)];
        elseif file == 2   % Samsung Babble DB
            a = [location_feature sprintf('up_de_babble_25dB_ss_train_%04d.raw', n)];
            b = [location_target_1 sprintf('25dB_ss_train_%04d.raw', n)];
            c = [location_target_2 sprintf('25dB_ss_train_%04d.raw', n)];
            d = [location_target_3 sprintf('25dB_ss_train_%04d.raw', n)];
            e = [location_target_4 sprintf('25dB_ss_train_%04d.raw', n)];
        elseif file == 3   % NTT Clean DB
            a = [location_feature sprintf('up_de_25dB_ntt_train_%04d.raw', n)];
            b = [location_target_1 sprintf('25dB_ntt_train_%04d.raw', n)];
            c = [location_target_2 sprintf('25dB_ntt_train_%04d.raw', n)];
            d = [location_target_3 sprintf('25dB_ntt_train_%04d.raw', n)];
            e = [location_target_4 sprintf('25dB_ntt_train_%04d.raw', n)];
        elseif file == 4   % NTT Babble DB
            a = [location_feature sprintf('up_de_babble_25dB_ntt_train_%04d.raw', n)];
            b = [location_target_1 sprintf('25dB_ntt_train_%04d.raw', n)];
            c = [location_target_2 sprintf('25dB_ntt_train_%04d.raw', n)];
            d = [location_target_3 sprintf('25dB_ntt_train_%04d.raw', n)];
            e = [location_target_4 sprintf('25dB_ntt_train_%04d.raw', n)];
        elseif file == 5   % ETRI Clean DB
            a = [location_feature sprintf('up_de_25dB_etri_train_%04d.raw', n)];
            b = [location_target_1 sprintf('25dB_etri_train_%04d.raw', n)];
            c = [location_target_2 sprintf('25dB_etri_train_%04d.raw', n)];
            d = [location_target_3 sprintf('25dB_etri_train_%04d.raw', n)];
            e = [location_target_4 sprintf('25dB_etri_train_%04d.raw', n)];
        elseif file == 6   % ETRI Babble DB
            a = [location_feature sprintf('up_de_babble_25dB_etri_train_%04d.raw', n)];
            b = [location_target_1 sprintf('25dB_etri_train_%04d.raw', n)];
            c = [location_target_2 sprintf('25dB_etri_train_%04d.raw', n)];
            d = [location_target_3 sprintf('25dB_etri_train_%04d.raw', n)];
            e = [location_target_4 sprintf('25dB_etri_train_%04d.raw', n)];
        else
            disp('ERROR');
            pause;
        end
        % output file
        out = [output sprintf('HBWE_clean_%d.raw', n)];  	% DNN output

        fid1 = fopen(a, 'rb'); 
        fid2 = fopen(b, 'rb');
        %fid3 = fopen(c, 'rb');
        %fid4 = fopen(d, 'rb');
        %fid5 = fopen(e, 'rb');
        signal1 = fread(fid1, inf, 'short'); signal1 = signal1(1+596:end, 1);
        signal2 = fread(fid2, inf, 'short');
        %signal3 = fread(fid3, inf, 'short');
        %signal4 = fread(fid4, inf, 'short');
        %signal5 = fread(fid5, inf, 'short');
        
        signal2 = signal2(1:length(signal1));
        %signal3 = signal3(1:length(signal1));
        %signal4 = signal4(1:length(signal1));
        %signal5 = signal5(1:length(signal1));
        fclose(fid1); fclose(fid2); %fclose(fid3); fclose(fid4); fclose(fid5);

        alpha = 1;
        prev_est_wbmag = zeros(1, 32);
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

        vm_result = zeros(Framenum, 1);
        VAD = zeros(Framenum, 1);
        frame_energy = zeros(Framenum, 1);
        noisy = zeros(Framenum, 1);
        Sph_Out = zeros(framelen2, 1);
        Sph_Prev = zeros(nfft2, 1);
        Sph_FE_Out = zeros(length(signal1), 1);
        ch_energy = zeros(1, 16);

        %% Bandwidth extension routine
        frm_cnt=0;
        for i = 1 : Framenum
            levelpara = zeros(1, 32);
            Sph_Frm1 = zeros(512, 1);
            Sph_Frm2 = zeros(512, 1);
            Sph_Frm3 = zeros(512, 1);
            Sph_Frm4 = zeros(512, 1);
            Sph_Frm5 = zeros(512, 1);
            if(i >= 2)
                Sph_Frm1(1 : overlap2) = signal1((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
                Sph_Frm2(1 : overlap2) = signal2((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
                %Sph_Frm3(1 : overlap2) = signal3((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
                %Sph_Frm4(1 : overlap2) = signal4((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
                %Sph_Frm5(1 : overlap2) = signal5((i - 1) * framelen2 - overlap2  + 1 : (i - 1) * framelen2);
            end
            Sph_Frm1(overlap2 + 1 : overlap2 + framelen2) = signal1((i - 1) * framelen2 + 1 : i * framelen2);
            Sph_Frm1 = Sph_Frm1 .* Window_w;

            Sph_Frm2(overlap2 + 1 : overlap2 + framelen2) = signal2((i - 1) * framelen2 + 1 : i * framelen2);
            Sph_Frm2 = Sph_Frm2 .* Window_w;
            
            %Sph_Frm3(overlap2 + 1 : overlap2 + framelen2) = signal3((i - 1) * framelen2 + 1 : i * framelen2);
            %Sph_Frm3 = Sph_Frm3 .* Window_w;
            
            %Sph_Frm4(overlap2 + 1 : overlap2 + framelen2) = signal4((i - 1) * framelen2 + 1 : i * framelen2);
            %Sph_Frm4 = Sph_Frm4 .* Window_w;
            
            %Sph_Frm5(overlap2 + 1 : overlap2 + framelen2) = signal5((i - 1) * framelen2 + 1 : i * framelen2);
            %Sph_Frm5 = Sph_Frm5 .* Window_w;

            nbfft = fft(Sph_Frm1, 512);
            target_fft_2 = abs(fft(Sph_Frm2, 512));
            %target_fft_3 = abs(fft(Sph_Frm3, 512));
            %target_fft_4 = abs(fft(Sph_Frm4, 512));
            %target_fft_5 = abs(fft(Sph_Frm5, 512));

            if i > 1
                alpha_ch = 0.55;
            else
                alpha_ch = 1;
            end

            % VAD
            vm = 0;
            for ch = 1 : 16
                j1 = ch_tbl(ch, 1); j2 = ch_tbl(ch, 2);
                energy = 0;
                for jj = j1 : j2
                    energy = energy + abs(nbfft(jj)).^2;
                end
                energy = energy / (j2 - j1 + 1);
                ch_energy(ch) = max((1 - alpha_ch) * ch_energy(ch) + alpha_ch * energy, min_energy);
                if i < 3
                    noise_energy(ch) = max(ch_energy(ch), 16);
                end
                snr(ch) = round(10 * log10((ch_energy(ch) / noise_energy(ch)) / 0.375));
                jj = max(min(snr(ch), 90), 1);
                vm = vm + vm_tbl(jj);
            end
            vm_result(i) = vm;

            nbfeature = abs(nbfft(1:257, 1));
            nbfeatures(i, :) = log((nbfeature(2 : 128, 1).^2).');
            nbfeatures_new = Features_16(Sph_Frm1, nbfeature(1 : 129).^2, nfft2);

            if (min(nbfeatures(i,:))==-inf||min(target_fft_2)==0)
                %min(target_fft_3)==0||min(target_fft_4)==0||min(target_fft_5)==0)
                continue;
            else
                nbphase = angle(nbfft);

                % log power magnitude extension using DNN
                % Normal DNN
                buf1 = bsxfun(@minus, nbfeatures(i, :), normal_mean_f);
                data = bsxfun(@rdivide, buf1, normal_std_f);
                w1probs = max((data * w1_normal + b1_normal.'), 0);
                w2probs = max((w1probs * w2_normal + b2_normal.'), 0);
                w3probs = max((w2probs * w3_normal + b3_normal.'), 0);
                est_l1 = w3probs * w4_normal + b4_normal.';
                buf1 = bsxfun(@times, est_l1, normal_std_t);
                est_l1 = bsxfun(@plus, buf1, normal_mean_t);

                feature=[nbfeatures(i, :) nbfeatures_new est_l1];

                % Classification 
                buf1 = bsxfun(@minus, feature, class_mean_f);
                data = bsxfun(@rdivide, buf1, class_std_f);
                w1probs = max((data * w1_class + b1_class.'), 0);
                w2probs = max((w1probs * w2_class + b2_class.'), 0);
                wclass(i, :) = exp(w2probs * w3_class + b3_class.');
                wclass(i, :) = wclass(i, :) ./ repmat(sum(wclass(i,:), 2), 1, size(wclass(i, :), 2));

                % Over-est. DNN
                buf1 = bsxfun(@minus, nbfeatures(i, :), over_mean_f);
                data = bsxfun(@rdivide, buf1, over_std_f);
                w1probs = max((data * w1_over + b1_over.'), 0);
                w2probs = max((w1probs * w2_over + b2_over.'), 0);
                w3probs = max((w2probs * w3_over + b3_over.'), 0);
                est_l2 = w3probs * w4_over + b4_over.';
                buf1 = bsxfun(@times, est_l2, over_std_t);
                est_l2 = bsxfun(@plus, buf1, over_mean_t);

                est_l = est_l1 * wclass(i, 1) + est_l2 * wclass(i, 2);

                % Spectral Folding
                est_wbmag_l = [nbfeature(1 : 129).' nbfeature(128 : -1 : 2).' nbfeature(257).'];
                % Spectral Smoothing
                for k=177:256
                    est_wbmag_l(k) = 0.2 * est_wbmag_l(k) + 0.7 * est_wbmag_l(k-1);
                end
                est_wbmag_l=log(est_wbmag_l.^2);

                data_band = est_l;

                for k = 1 : 32
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

                est_l = est_l + levelpara;

                for k = 1 : 32
                    band_energy = mean(est_wbmag_l(129 + 4 * (k - 1) : 132 + 4 * (k - 1))) - est_l(1, k);
                    est_wbmag_l(129 + 4 * (k - 1) : 132 + 4 * (k - 1)) = est_wbmag_l(129 + 4 * (k - 1) : 132 + 4 * (k - 1)) - band_energy;
                end

                for k = 1 : 31
                    for fb = 1 : 4
                        thres = (est_l(1, k+1) - est_l(1, k)) / 4 * (fb - 1) + est_l(1, k);
                        if thres >= 0
                            if est_wbmag_l(128 + fb + 4 * (k - 1)) > 1.05 * thres
                                est_wbmag_l(128 + fb + 4 * (k - 1)) = 1.05 * thres;
                            end
                        else
                            if est_wbmag_l(128 + fb + 4 * (k - 1)) > 0.95 * thres
                                est_wbmag_l(128 + fb + 4 * (k - 1)) = 0.95 * thres;
                            end
                        end
                    end
                end

                for fb = 1 : 4
                    thres = (est_l(1, 32) - est_l(1, 31)) / 4 * (fb + 3) + est_l(1, 31);
                    if thres >= 0
                        if est_wbmag_l(252 + fb) > 1.05 * thres
                            est_wbmag_l(252 + fb) = 1.05 * thres;
                        end
                    else
                        if est_wbmag_l(252 + fb) > 0.95 * thres
                            est_wbmag_l(252 + fb) = 0.95 * thres;
                        end
                    end
                end

                est_wbmag_l = exp(0.5 * est_wbmag_l);

                % VAD
                if i > 8
                    if vm_result(i) < 205
                        silence = silence + 1;
                        if silence >= 14
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
                    if frame_energy(i) <= 4.2558e+04
                        case1 = case1 + 1;
                        case2 = 0;
                        if case1 >= 4
                            noisy(i) = 0;
                        end
                        if case1 == 1
                            est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 1;
                            est_wbmag_l(129 : 256) = est_wbmag_l(129 : 256) * 0.5;
                        elseif case1 == 2
                            est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.5;
                            est_wbmag_l(129 : 256) = est_wbmag_l(129 : 256) * 0.25;
                        elseif case1 == 3
                            est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.25;
                            est_wbmag_l(129 : 256) = est_wbmag_l(129 : 256) * 0.125;
                        elseif case1 >= 4
                            est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 0.125;
                            est_wbmag_l(129 : 256) = est_wbmag_l(129 : 256) * 0.0625;
                        end
                    else
                        case1 = 0;
                        case2 = case2 + 1;
                        if case2 >= 4
                            noisy(i) = 1;
                        end
                        est_wbmag_l(2 : 128) = est_wbmag_l(2 : 128) * 1;
                        est_wbmag_l(129 : 256) = est_wbmag_l(129 : 256) * 0.125;
                    end
                else
                    noisy(i) = noisy(i-1);
                    est_wbmag_l = 20 * log10(est_wbmag_l);

                    buf = est_wbmag_l;
                    for fb = 130 : 255
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
                        his_thres(fb) = 52;
                    end
                    for fb = 5 : 10
                        his_thres(fb) = 52 - (fb - 4);
                    end
                    for fb = 11 : 256
                        his_thres(fb) = 46 - 0.0255 * (fb - 10);
                    end

                    for fb = 2 : 128
                        if est_wbmag_l(fb) < his_thres(fb)
                            est_wbmag_l(fb) = est_wbmag_l(fb) - 6;
                        end
                    end
                    for fb = 129 : 256
                        if est_wbmag_l(fb) < his_thres(fb)
                            est_wbmag_l(fb) = est_wbmag_l(fb) - 12;
                        end
                    end

                    est_wbmag_l = 10 .^ ((est_wbmag_l) / 20);
                end

                est_wbmag_r = fliplr(est_wbmag_l(1, 2 : 256));
                est_wbmag = [est_wbmag_l est_wbmag_r];

                wbphase_l = [nbphase(1 : 129, 1).' -nbphase(128 : -1 : 1, 1).'];
                wbphase_r = fliplr(wbphase_l(1, 2 : 256));
                wbphase = [wbphase_l -wbphase_r];

                Sph_FFT = est_wbmag.' .* exp(1i * wbphase.');
                frm_cnt = frm_cnt+1;
                
                % Over-est. Cost Calc.
                test_out(frm_cnt,:)    = 2*log( est_wbmag(1:256).'  );
                target_out_2(frm_cnt,:)  = 2*log( target_fft_2(1:256) );  % 7000
                %target_out_3(frm_cnt,:)  = 2*log( target_fft_3(1:256) );  % 6750
                %target_out_4(frm_cnt,:)  = 2*log( target_fft_4(1:256) );  % 6500
                %target_out_5(frm_cnt,:)  = 2*log( target_fft_5(1:256) );  % 6250
                
                if ( min(nbfeatures(i,:)) == -inf )
                    nbfeatures(i,:) = ones(1, size(nbfeatures(i,:),1))*eps;
                end
                
            end  % XOR

            
        end  % FRAME


        % Segmentation
        start_f = [];
        end_f   = [];
        seg_idx=[];
        for kk = 2 : Framenum
            if ( VAD(kk) == 1 && VAD(kk-1) == 0 )
               start_f = [start_f kk];
            elseif ( VAD(kk) == 0 && VAD(kk-1) == 1 )
               end_f = [end_f kk];
            end
        end
        if length(start_f) > length(end_f)
            end_f = [end_f Framenum];
        elseif length(start_f) < length(end_f)
            start_f = [1 start_f];
        end

        for kk = 1 : length(start_f)
            seg_idx = [seg_idx; start_f(kk) end_f(kk)];
        end 
        cost_1=[];
        cost_2=[];
        Final_cost=[];

        for k = 1:size(seg_idx, 1)
            for kk = seg_idx(k,1) : seg_idx(k,2)
                cost_1(kk,:) = (test_out(kk,:) - target_out_2(kk,:));
                cnt = 0;
                for kkk= 1:256
                    if cost_1(kk, kkk) > 0
                        cost_2(kk,kkk) = cost_1(kk, kkk);
                        cnt=cnt+1;
                    else
                        cost_2(kk,kkk) = 0;
                    end
                end

            end
        end
        Final_cost = sum(cost_2,2);  % ���� ����ϴ� Cost
        
        over_frm_idx_1 = find( (Final_cost > 100) );

        % Segment���� over-est. ���� �Ǵ�. 
        over_index=[]; 
        count = zeros(size(seg_idx, 1),1);
        for k = 1:size(seg_idx, 1)
            for kk = 1:size(over_frm_idx_1, 1)
                if ( (over_frm_idx_1(kk) >= seg_idx(k,1)) && ((over_frm_idx_1(kk) <= seg_idx(k,2))) )
                    count(k) = count(k) + 1;   % Count�� 0�� ��� �ش� segment�� over-est. �� ����.                 
                end
            end
        end

        % over-est. ���� ���� (segment ����)
        over_nbfeature_temp=[];  over_nbfeature_temp_file=[];
        over_target_temp=[];     over_target_temp_file=[];
        for k = 1: size(seg_idx, 1)
            if ( count(k) < 30 )     % continue
                over_nbfeature_temp = -inf;
                over_target_temp    = -inf;
                cnt_1 = cnt_1+1;
            elseif ( count(k) < 70 )     % 7000 (cnt_2)
                over_nbfeature_temp = nbfeatures(seg_idx(k,1):seg_idx(k,2), :);
                over_target_temp = target_out_2(seg_idx(k,1):seg_idx(k,2), :);
                cnt_2  = cnt_2+1;
            elseif( count(k) < 150 )  % 6750 (cnt_3)
                over_nbfeature_temp = nbfeatures(seg_idx(k,1):seg_idx(k,2), :);
                over_target_temp = target_out_3(seg_idx(k,1):seg_idx(k,2), :);
                cnt_3  = cnt_3+1;
            elseif( count(k) < 180 ) % 6500 (cnt_4)
                over_nbfeature_temp = nbfeatures(seg_idx(k,1):seg_idx(k,2), :);
                over_target_temp = target_out_4(seg_idx(k,1):seg_idx(k,2), :);
                cnt_4  = cnt_4+1;
            else                     % 6250 (cnt_5)
                over_nbfeature_temp = nbfeatures(seg_idx(k,1):seg_idx(k,2), :);
                over_target_temp = target_out_5(seg_idx(k,1):seg_idx(k,2), :);
                cnt_5  = cnt_5+1;
            end
            % ���� 1���� ���� Feauture
            if ( (min(min(over_nbfeature_temp)) ~= -inf) && (min(min(over_target_temp)) ~= -inf) )
                over_nbfeature_temp_file = [over_nbfeature_temp_file; over_nbfeature_temp];
                over_target_temp_file    = [over_target_temp_file; over_target_temp];
            end
        end
        % DB 1���� ���� Feauture
        over_nbfeature_temp_DB    = [over_nbfeature_temp_DB; over_nbfeature_temp_file];
        over_target_temp_DB       = [over_target_temp_DB; over_target_temp_file];
    end  % ���� 1��
      
    % ��ü DB�� ���� Feature
    over_nbfeature_temp_total   = [over_nbfeature_temp_total; over_nbfeature_temp_DB];
    over_target_temp_total      = [over_target_temp_total; over_target_temp_DB];
    
    over_target=[];
    for kkk = 1:32
        over_target(:, kkk) = mean( over_target_temp_total(:, 4*kkk-3:4*kkk), 2 );
    end
    
    disp('Normalize...');
    % Normalize
    over_std_f = single(std(over_nbfeature_temp_total));
    over_mean_f = single(mean(over_nbfeature_temp_total));

    over_mean_t = single(mean(over_target));
    over_std_t = single(std(over_target));

    buf1 = bsxfun(@minus, over_nbfeature_temp_total, over_mean_f);
    over_nbfeature = bsxfun(@rdivide, buf1, over_std_f);
    clear over_nbfeature_temp_total;

    buf2 = bsxfun(@minus, over_target, over_mean_t);
    over_target = bsxfun(@rdivide, buf2, over_std_t);

    totnum=size(over_nbfeature,1);
    rand('state',0); %so we know the permutation of the training data
    randomorder=randperm(totnum);

    over_nbfeature = over_nbfeature(randomorder(1:end),:);
    over_target = over_target(randomorder(1:end),:);

    disp('Save...');

    over_nbfeature = single(over_nbfeature);
    over_target    = single(over_target);
    
    save(sprintf('train_over_%d.mat', file), 'over_nbfeature', 'over_target');
    save(sprintf('norm_over_%d.mat', file), 'over_mean_f', 'over_std_f', 'over_mean_t', 'over_std_t');    
    
    
end  % DB 1��


fclose('all');
disp('FINISH');









