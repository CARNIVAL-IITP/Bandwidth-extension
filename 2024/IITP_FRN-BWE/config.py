class CONFIG:
    gpus = "0"  # List of gpu devices

    class TASK:
        task = 'HB-BWE'  # Task to execute. Should either be 'msm' or 'bwe'
        assert task in ['MSM-clean', 'MSM-noisy','NAE', 'NB-BWE' ,'NB-BWE+MSM','HB-BWE', 'PLC'], "task should either be 'msm' or 'bwe'"

        mask_chunk = 256  # Size of masked chunks for MSM. Should be a power of two
        mask_ratio = 0.5  # MSM masking ratio in range (0, 1)

        masking_strategy = 'structured'
        assert masking_strategy in ['structured', 'unstructured']
        unstructured_mask_chunk = [64, 128, 256, 512, 1024]
        unstructed_mask_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        '''
        BWE downsampling method. Should be either 'cheby', 'augment' or resampy supported methods.
            'cheby' uses the Scipy's decimation based on the Chebyshev Type-I lowpass filter.
            'augment' uses the Chebyshev Type-I filters with random orders and ripples.
        '''
        downsampling = 'cheby'

        # resampy supported methods
        resampy = ['kaiser_best', 'kaiser_fast', 'fft', 'polyphase', 'linear', 'zero_order_hold', 'sinc_best',
                   'sinc_medium', 'sinc_fastest', 'soxr_vhq', 'soxr_hq', 'soxr_mq', 'soxr_lq', 'soxr_qq']
        assert downsampling in ['augment', 'cheby'] + resampy, 'Invalid downsampling method'
        orders = [8]#range(1, 11)  # the Chebyshev Type-I orders
        ripples = [0.05]#[1e-9, 1e-6, 1e-3, 1, 5]  # the Chebyshev Type-I ripples

    class TRAIN:
        batch_size = 64  # number of audio files per batch
        lr = 1e-4  # learning rate
        epochs = 100  # max training epochs
        workers = 16  # number of dataloader workers
        val_split = 0.1  # validation set proportion
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        pretraining = False

        # loss_type = 3  # training loss types. 1: MSE loss, 2: MSE and multi-resolution STFT loss
        # assert loss_type in [1, 2, 3, 4, 5], '1: time only, 2: time + stft, 3: time + stft + subband'
        # mse_weight = 10000  # weight of the MSE loss
        # stft_weight = 1  # weight of the MSE loss
        
        # class subband:
        #     subband_training = True
        #     subband = 8
        #     weight_loss = 1

        class subband:
            subband_training = True
            subband = 1
            weight_loss_1 = 1
            weight_loss_2 = 0
            weight_loss_4 = 0
            weight_loss_8 = 0

        # class pretraining:
        #     strategy = 'none'
        #     regularizer_mode = False
        #     regularizer = 'none'
        #     lambda_reg = 0.001
        #     ewc_mode = False
        #     ewc_lambda = 0.0001
        #     ema_mode = False
        #     ema_decay = 0.9999
        #     gem_mode = False
        #     memory_strength = 0.00001
        #     lr0 = 3e-4
        #     num_prior_training = 1 


    # # Model config
    # class MODEL:
    #     model_name = 'FRN-continual-baseline'
    #     assert model_name in ['FRN-baseline', 'FRN-FiLM', 'FRN-encoder', 'FRN-continual-baseline']
    #     state = True
    #     enc_lstm_tpye = 'LT-LSTM'
    #     assert enc_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
    #     enc_layers = 4  # number of MLP blocks in the encoder
    #     enc_in_dim = 384  # dimension of the input projection layer in the encoder
    #     enc_dim = 768  # dimension of the MLP blocks
    #     pred_lstm_tpye = 'LT-LSTM'
    #     assert pred_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
    #     pred_dim = 512  # dimension of the LSTM in the predictor
    #     pred_layers = 1  # number of LSTM layers in the predictor

    # Model config
    class MODEL:
        model_name = 'FRN-subband-pcl'
        # model_name = 'FRN-baseline'
        # model_name = 'FRN-baseline-continual'
        # model_name = 'FRN-subband'
        assert model_name in ['FRN-baseline', 'FRN-FiLM', 'FRN-encoder', 'FRN-baseline-continual', 'FRN-subband', 'FRN-subband-pcl']
        enc_state = True
        pred_state = True
        enc_lstm_tpye = 'LSTM'
        assert enc_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_lstm_tpye = 'LSTM'
        assert pred_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
        pred_dim = 512  # dimension of the LSTM in the predictor
        pred_layers = 1  # number of LSTM layers in the predictor


    # Dataset config
    class DATA:
        dataset = 'sitec-rir-each'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'timit': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/TIMIT',
                             'train': '/media/dh2/Datasets_linux1/TUNet_dataset/TIMIT/train.txt',
                             'test': '/media/dh2/Datasets_linux1/TUNet_dataset/TIMIT/test.txt'},
                    'vctk-0.92-multi': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92',
                             'train': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92/vctk-0.92-multi_train_wavs.txt",
                             'test': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92/vctk-0.92-multi_test_wavs.txt"},
                    'vctk-0.92-single': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92',
                             'train': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92/vctk-0.92-single_train_wavs.txt",
                             'test': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.92/vctk-0.92-single_test_wavs.txt"},
                    'vctk-0.80-multi': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/vctk/wav48',
                             'train': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.80/train.txt",
                             'test': "/media/dh2/Datasets_linux1/TUNet_dataset/vctk-0.80/test.txt"},
                    'sitec-rir-each': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/sitec_rir',
                             'train': "/media/dh2/Datasets_linux1/TUNet_dataset/sitec_rir/sitec_rir_each_tr.txt",
                             'val': "/media/dh2/Datasets_linux1/TUNet_dataset/sitec_rir/sitec_rir_each_val.txt",
                             'test': "/media/dh2/Datasets_linux1/TUNet_dataset/sitec_rir/sitec_rir_each_test.txt"},
                    'plc-challenge': {'root': '/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge',
                            'train_clean': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/train_clean.txt",
                            'train_noisy': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/train_noisy.txt",
                            'val_clean': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/val_clean.txt",
                            'val_noisy': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/val_noisy.txt",
                            'test_clean': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/test_clean.txt",
                            'test_noisy': "/media/dh2/Datasets_linux1/TUNet_dataset/plc-challenge/test_noisy.txt"}
                    }
        sr = 16000  # audio sampling rate
        audio_chunk_len = 40960  # size of chunk taken in each audio files
        window_size = 320  # window size of the STFT operation, equivalent to packet size
        stride = 160  # stride of the STFT operation
        ratio = 2

        class TRAIN:
            packet_sizes = [256, 512, 768, 960, 1024,
                            1536]  # packet sizes for training. All sizes should be divisible by 'audio_chunk_len'
            transition_probs = ((0.9, 0.1), (0.5, 0.1), (0.5, 0.5))  # list of trainsition probs for Markow Chain

        class EVAL:
            packet_size = 320  # 20ms
            transition_probs = [(0.9, 0.1)]  # (0.9, 0.1) ~ 10%; (0.8, 0.2) ~ 20%; (0.6, 0.4) ~ 40%
            masking = 'gen'  # whether using simulation or real traces from Microsoft to generate masks
            assert masking in ['gen', 'real']
            trace_path = 'test_samples/blind/lossy_singals'  # must be clarified if masking = 'real'

    class LOG:
        log_dir = 'lightning_logs'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.

    class TEST:
        in_dir = '/home/dh2/Project/IITP/2023/STOI/noisy'  # path to test audio inputs
        out_dir = '/home/dh2/Project/IITP/2023/STOI/proposed_FRN-subband-pcl/'  # path to generated outputs
        save = True
