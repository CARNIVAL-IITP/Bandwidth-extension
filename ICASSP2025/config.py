class CONFIG:
    gpus = "0"  # List of gpu devices

    # Bandwidth extension and masked speech modeling experiment config
    class TASK:
        task = 'HB-BWE'  # Task to execute. Should either be 'msm' or 'bwe'
        assert task in ['MSM', 'NAE', 'NB-BWE' ,'NB-BWE+MSM','HB-BWE'], "task should either be 'msm' or 'bwe'"
        mask_chunk = 256  # Size of masked chunks for MSM. Should be a power of two
        mask_ratio = 0.2  # MSM masking ratio in range (0, 1)

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
        lr = 3e-4  # learning rate
        epochs = 100  # max fine-tuning epochs
        workers = 8  # number of dataloader workers
        val_split = 0.1  # validation set proportion    
        loss_type = 3  # training loss types. 1: MSE loss, 2: MSE and multi-resolution STFT loss
        assert loss_type in [1, 2, 3, 4, 5], '1: time only, 2: time + stft, 3: time + stft + subband, 4: time + subband'
        mse_weight = 10000  # weight of the MSE loss
        stft_weight_loss = 1  # weight of the MSE loss
        clipping_val = 1.0  # gradient clipping value
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        optimizer = 'adam'
        assert optimizer in ['sgd', 'adam', 'adamw' ,'adamax']
        momentum = 0.8

        class subband:
            subband_training = True
            subband = 8
            weight_loss = 1

        # class subband:
        #     subband_training = True
        #     subband = 8
        #     weight_loss_1 = 0
        #     weight_loss_2 = 0
        #     weight_loss_4 = 0
        #     weight_loss_8 = 1

        class consistency:
            consistency_training = False
            consistency_loss = 'L2'
            assert consistency_loss in ['L1', 'L2']
            concistency_weight_loss = 1

        class pretraining:
            pretrained = False
            num_prior_training = 2 
            pretrained_checkpoint = ''
            strategy = 'naive'
            assert strategy in ['naive', 'L1', 'L2' ,'EWC', 'EWC+','EMA','GEM']
            ewc_mode = False
            regularizer = 'none'
            assert regularizer in ['L1', 'L2', 'none']
            regularizer_weight = 0.0001
            # weight_decay = 0.001
            ema_decay = 0.99 
            lr0 = 3e-4


    # Model config
    class MODEL:
        # model_name = 'TUNet-baseline'
        model_name = 'AudioUNet_TFiLM'
        # model_name = 'AudioUNet'
        tfilm = True  # enable/disable TFiLM layers
        n_blocks = 64  # number of blocks of TFiLM layers.
        bottleneck_type = 'performer'  # bottleneck module. Should either be 'performer', 'lstm' or None
        assert bottleneck_type in ['performer', 'lstm', None], "Invalid bottleneck_type"
        kernel_sizes = [66, 18, 8]  # kernel sizes of each convolution/deconvolution layers
        strides = [4, 4, 4]  # strides of each conv olution/deconvolution layers
        out_channels = [64, 128, 256]  # numbers of filters of each convolution/deconvolution layers
        
        # Performer bottleneck config
        class TRANSFORMER:
            dim_head = 32
            depth = 3
            heads = 2

    # Dataset config
    class DATA:
        dataset = 'timit'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'timit': {'root': '/media/dh2/spteam2_dataset/TUNet_dataset/TIMIT',
                             'train': '/media/dh2/spteam2_dataset/TUNet_dataset/TIMIT/train.txt',
                             'test': '/media/dh2/spteam2_dataset/TUNet_dataset/TIMIT/test.txt'},
                    'vctk-0.92-multi-old': {'root': '/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92',
                             'train': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92-multi_train_wavs.txt",
                             'test': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92-multi_test_wavs.txt"},
                    'vctk-0.92-single': {'root': '/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92',
                             'train': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92-single_train_wavs.txt",
                             'test': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.92-single_test_wavs.txt"},
                    'vctk-0.92-multi': {'root': '/media/dh2/spteam2_dataset/TUNet_dataset/VCTK_LPF/8-16/vctk-0.92-multi',
                             'train': '/media/dh2/spteam2_dataset/TUNet_dataset/VCTK_LPF/8-16/vctk-0.92-multi/train.txt',
                             'valid': '/media/dh2/spteam2_dataset/TUNet_dataset/VCTK_LPF/8-16/vctk-0.92-multi/valid.txt',
                             'test': '/media/dh2/spteam2_dataset/TUNet_dataset/VCTK_LPF/8-16/vctk-0.92-multi/test.txt'},
                    'vctk-0.80-multi': {'root': '/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.80/wav48',
                             'train': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.80/train.txt",
                             'test': "/media/dh2/spteam2_dataset/TUNet_dataset/vctk-0.80/test.txt"}}
        assert dataset in data_dir.keys(), 'Unknown dataset.'
        sr = 16000  # target audio sampling rate
        ratio = 2  # downsampling ratio
        window_size = 8192  # size of the sliding window
        stride = 4096  # stride of the sliding window. Should be divisible to 'mask_chunk' if the task is MSM.

    class LOG:
        log_dir = 'lightning_logs/'  # checkpoint and log directory
        sample_path = 'audio_samples'  # path to save generated audio samples in evaluation.

    class TEST:
        # in_dir = '/home/dh2/Research/DB_TIMIT/test_lr/lpf_decimate/name/'  # path to test audio inputs
        in_dir = '/home/dh2/Research/DB_TIMIT/test_lr/lr_msm/'  # path to test audio inputs
        out_dir = '/home/dh2/Research/TUNet/TUNet-bwe-pretraining/output/'  # path to generated outputs