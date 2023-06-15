class Args:
    data = '../data'
    batch_size = 96
    learning_rate = 0.1
    learning_rate_min = 0.001
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = 0
    epochs = 1
    init_channels = 16
    layers = 8
    model_path = 'saved_models'
    cutout = False
    cutout_length = 16
    drop_path_prob = 0.3
    save = 'EXP'
    seed = 2
    grad_clip = 5
    train_portion = 0.5
    unrolled = False
    arch_learning_rate = 3e-4
    arch_weight_decay = 1e-3
    optimization = 'DARTS'
    arch_search_method = 'DARTS'
    lambda_train_regularizer = 1
    lambda_valid_regularizer = 1
    early_stopping = 0
