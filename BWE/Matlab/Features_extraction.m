%% Features extraction
% Speech/Acoustics/Audio Signal Processing Lab., Hanyang Univ., 2016
fclose('all'); close all; clear; clc;

for file = 1:18 %1:40 %1:64
    location_feature = './BWE_DB_new/LPF_UP_RAW/test/clean/';
    location_target = './BWE_DB_new/16K_RAW/test/clean/';
    
    input1 = []; input2 = []; target = [];
    Window_w = zeros(512, 1);
    
    % Initialize parameter & define window function
    framelen2 = 320; fs2 = 16000; nfft2 = 512; overlap2 = 80;
    Deno2 = 0.5 / overlap2;
    for k2 = 1:overlap2
        f_tmp2 = sin(pi*((k2 - 1) + 0.5) * Deno2);
        Window_w(k2) = f_tmp2 * f_tmp2;
    end
    for k2 = (overlap2 + 1):framelen2
        Window_w(k2) = 1.0;
    end
    for k2 = (framelen2 + 1):(framelen2 + overlap2)
        f_tmp2 = sin(pi * (((k2 - 1) - framelen2 + overlap2) + 0.5) * Deno2);
        Window_w(k2) = f_tmp2 * f_tmp2;
    end
    for k2 = (framelen2 + overlap2 + 1):nfft2
        Window_w(k2) = 0;
    end
    
 
    for n = 1+(file-1)*100:file*100
        disp(n);
        % Load file & delay calculation
        a = [location_feature sprintf('LPF_UP_RAW_test--%d.raw', n)];
        b = [location_target sprintf('16k_raw_test_clean-%d.raw', n)];
        fid1 = fopen(a, 'rb');
        signal1 = fread(fid1, inf, 'short');
        fclose(fid1);
        fid2 = fopen(b, 'rb');
        signal2 = fread(fid2, inf, 'short');
        fclose(fid2);
        d=finddelay(signal1, signal2);
        %signal1 = signal1(1+516:end, 1);
        
        % Parameters
        Framenum = floor(length(signal1)/framelen2);
        
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
            
            Sph_Frm1 = zeros(512, 1);
            if(i >= 2)
                Sph_Frm1(1:overlap2) = signal1((i - 1)*framelen2 - overlap2  + 1:(i - 1)*framelen2);
            end
            Sph_Frm1(overlap2 + 1:overlap2 + framelen2) = signal1((i - 1)*framelen2 + 1:i*framelen2);
            Sph_Frm1 = Sph_Frm1 .* Window_w;
            
            nbfft = fft(Sph_Frm1, 512);
            
            Sph_Frm2 = zeros(512, 1);
            if(i >= 2)
                Sph_Frm2(1:overlap2) = signal2((i - 1)*framelen2 - overlap2  + 1:(i - 1)*framelen2);
            end
            Sph_Frm2(overlap2 + 1:overlap2 + framelen2) = signal2((i - 1)*framelen2 + 1:i*framelen2);
            Sph_Frm2 = Sph_Frm2 .* Window_w;
            
            wbfft = fft(Sph_Frm2, 512);
            
            nbfeature(k, :) = log(abs(nbfft(1:129, 1)).^2);
            nbnewfeatures(k, :) = Features_16(Sph_Frm1, abs(nbfft(1:129, 1)).^2, nfft2);
            wbfeature(k, :) = log(abs(wbfft(1:257, 1)).^2);
            k=k+1;
        end
        
        nbfeature=nbfeature(:,2:129);
        wbfeature=wbfeature(:,129:257);
        
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
    save(sprintf('MATLAB/FFT64/FEATURE_EXTRACT/TEST/test_feature_%d.,mat', file), 'input1', 'input2', 'target');
end

fclose('all'); close all;