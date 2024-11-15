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
        batch_size = 16  # number of audio files per batch
        lr = 1e-4  # learning rate
        epochs = 100  # max training epochs
        workers = 16  # number of dataloader workers
        val_split = 0.1  # validation set proportion
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        pretraining = False

    # Model config
    class MODEL:
        model_name = 'FRN-continual-baseline'
        assert model_name in ['FRN-baseline', 'FRN-FiLM', 'FRN-encoder', 'FRN-continual-baseline']
        state = True
        enc_lstm_tpye = 'LT-LSTM'
        assert enc_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
        enc_layers = 4  # number of MLP blocks in the encoder
        enc_in_dim = 384  # dimension of the input projection layer in the encoder
        enc_dim = 768  # dimension of the MLP blocks
        pred_lstm_tpye = 'LT-LSTM'
        assert pred_lstm_tpye in ['LSTM', 'LT-LSTM', 'GRU']
        pred_dim = 512  # dimension of the LSTM in the predictor
        pred_layers = 1  # number of LSTM layers in the predictor

    # Dataset config
    class DATA:
        dataset = 'vctk-0.92-multi'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'timit': {'root': '/home/dh2/Research/TUNet/data/TIMIT',
                             'train': '/home/dh2/Research/TUNet/data/TIMIT/train.txt',
                             'test': '/home/dh2/Research/TUNet/data/TIMIT/test.txt'},
                    'vctk-0.92-multi': {'root': '/home/dh2/Research/TUNet/data/vctk-0.92',
                             'train': "/home/dh2/Research/TUNet/data/vctk-0.92/vctk-0.92-multi_train_wavs.txt",
                             'test': "/home/dh2/Research/TUNet/data/vctk-0.92/vctk-0.92-multi_test_wavs.txt"},
                    'vctk-0.92-single': {'root': '/home/dh2/Research/TUNet/data/vctk-0.92',
                             'train': "/home/dh2/Research/TUNet/data/vctk-0.92/vctk-0.92-single_train_wavs.txt",
                             'test': "/home/dh2/Research/TUNet/data/vctk-0.92/vctk-0.92-single_test_wavs.txt"},
                    'vctk-0.80-multi': {'root': '/home/dh2/Research/TUNet/data/vctk/wav48',
                             'train': "/home/dh2/Research/TUNet/data/vctk-0.80/train.txt",
                             'test': "/home/dh2/Research/TUNet/data/vctk-0.80/test.txt"},
                    'sitec-rir-each': {'root': '/home/dh2/Research/TUNet/data/sitec_rir',
                             'train': "/home/dh2/Research/TUNet/data/sitec_rir/sitec_rir_each_tr.txt",
                             'test': "/home/dh2/Research/TUNet/data/sitec_rir/sitec_rir_each_test.txt"},
                    'plc-challenge': {'root': '/home/dh2/Research/TUNet/data/plc-challenge',
                            'train_clean': "/home/dh2/Research/TUNet/data/plc-challenge/train_clean.txt",
                            'train_noisy': "/home/dh2/Research/TUNet/data/plc-challenge/train_noisy.txt",
                            'val_clean': "/home/dh2/Research/TUNet/data/plc-challenge/val_clean.txt",
                            'val_noisy': "/home/dh2/Research/TUNet/data/plc-challenge/val_noisy.txt",
                            'test_clean': "/home/dh2/Research/TUNet/data/plc-challenge/test_clean.txt",
                            'test_noisy': "/home/dh2/Research/TUNet/data/plc-challenge/test_noisy.txt"}
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
        in_dir = '/home/dh2/Project/IITP/STOI/clean'  # path to test audio inputs
        out_dir = '/home/dh2/Project/IITP/STOI/clean-proposed_FRN-CLASS'  # path to generated outputs
        save = True
