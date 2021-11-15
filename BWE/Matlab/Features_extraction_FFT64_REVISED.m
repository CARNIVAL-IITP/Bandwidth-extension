%% Features extraction
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016
fclose('all'); close all; clear; clc;

for file = 1:40
    
    location_feature = 'BWE_DB/FFT64/FEATURE/TEST/CLEAN/'; 
    location_target = 'BWE_DB/FFT64/TARGET/TEST/CLEAN/';
    %dataset1
    %location_feature = './BWE_DB/FEATURE/LPF_16K_RAW/TRAIN1/CLEAN/';
    %location_target = './BWE_DB/TARGET/ORIGINAL_16K_RAW/TRAIN1/CLEAN/';
    %dataset2
    %location_feature = './BWE_DB/FEATURE/LPF_16K_RAW/TRAIN2/CLEAN/';
    %location_target = './BWE_DB/TARGET/ORIGINAL_16K_RAW/TRAIN2/CLEAN/';
    
    input1 = []; input2 = []; target = [];
    
    % Initialize parameter & define window function
    fs = 16000; framelen = 32; overlap = 32; nfft = 64;
    
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
    
    
    for n = 1+(file-1)*100:file*100
        disp(n);
        % Load file & delay calculation
        a = [location_feature sprintf('LPF_UP_RAW_test--%d.raw', n)];
        b = [location_target sprintf('16k_raw_test-%d.raw', n)];
        %a = [location_feature sprintf('lpf_up_raw_train1_clean-%d.raw', n)];
        %b = [location_target sprintf('16k_raw_train_clean-%d.raw', n)];
       % a = [location_feature sprintf('lpf_up_raw_train2_clean-%d.raw', n)];
        %b = [location_target sprintf('16k_raw_train2_clean-%d.raw', n)];
        fid1 = fopen(a, 'rb');
        signal1 = fread(fid1, inf, 'short');
        fclose(fid1);
        fid2 = fopen(b, 'rb');
        signal2 = fread(fid2, inf, 'short');
        fclose(fid2);
%         d=finddelay(signal1, signal2);
%         signal1 = signal1(1+516:end, 1);
        
        % Parameters
        Framenum = floor(length(signal1)/framelen);
        
        % LPS extraction
        nbfeature=[];
        nbnewfeatures=[];
        wbfeature=[];
        nbfeature_new=[];
        nbnewfeatures_new=[];
        wbfeature_new=[];
        
        k=1;
        
		% Extract log power magnitudes & Features
        for i = 1 : Framenum
            
            Sph_Frm1 = zeros(length(window), 1);
            if(i >= 2)
                Sph_Frm1(1:overlap) = signal1((i - 1)*framelen - overlap  + 1:(i - 1)*framelen);
            end
            Sph_Frm1(overlap + 1:overlap + framelen) = signal1((i - 1)*framelen + 1:i*framelen);
            Sph_Frm1 = Sph_Frm1 .* window;
            
            nbfft = fft(Sph_Frm1, nfft);
            
            Sph_Frm2 = zeros(length(window), 1);
            if(i >= 2)
                Sph_Frm2(1:overlap) = signal2((i - 1)*framelen - overlap  + 1:(i - 1)*framelen);
            end
            Sph_Frm2(overlap + 1:overlap + framelen) = signal2((i - 1)*framelen + 1:i*framelen);
            Sph_Frm2 = Sph_Frm2 .* window;
            
            wbfft = fft(Sph_Frm2, nfft);
            
            nbfeature(k, :) = log(abs(nbfft(1:(nfft/4+1), 1)).^2);
            nbnewfeatures(k, :) = Features(Sph_Frm1, abs(nbfft(1:(nfft/4+1), 1)).^2, nfft);
            wbfeature(k, :) = log(abs(wbfft((1:nfft/2+1), 1)).^2);
            k=k+1;
        end
        
        nbfeature=nbfeature(:,2:nfft/4);
        wbfeature=wbfeature(:,(nfft/4+1):nfft/2);
        
        j=0;
        
		% save except '-inf'
		
        for i = 1 : size(nbfeature, 1)
            if min(min(nbfeature(i,:)))==-inf || min(min(wbfeature(i,:)))==-inf
                nbfeature_new=[nbfeature_new; nbfeature(j+1:i-1,:)];
                nbnewfeatures_new=[nbnewfeatures_new; nbnewfeatures(j+1:i-1,:)];
                wbfeature_new=[wbfeature_new; wbfeature(j+1:i-1,:)];
                j=i;
            end
            if i==size(nbfeature, 1)
                nbfeature_new=[nbfeature_new; nbfeature(j+1:size(nbfeature, 1),:)];
                nbnewfeatures_new=[nbnewfeatures_new; nbnewfeatures(j+1:size(nbnewfeatures, 1),:)];
                wbfeature_new=[wbfeature_new; wbfeature(j+1:size(wbfeature, 1),:)];
            end
        end
        
        input1 = [input1; nbfeature_new];
        input2 = [input2; nbnewfeatures_new];
        target = [target; wbfeature_new];
    end
    
    %save(sprintf('MATLAB/FFT64/FEATURE_EXTRACT/TEST/test_feature_%d.mat', file), 'input1', 'input2', 'target');
    %save(sprintf('MATLAB/FFT64/FEATURE_EXTRACT/TRAIN1/train1_feature_%d.mat', file), 'input1', 'input2', 'target');
    save(sprintf('MATLAB/FFT64/FEATURE_EXTRACT/TRAIN2/train2_feature_%d.mat', file), 'input1', 'input2', 'target');
end

fclose('all'); close all;