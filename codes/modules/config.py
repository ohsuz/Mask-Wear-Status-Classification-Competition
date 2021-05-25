class Config:
    train_dir = '/opt/ml/input/data/train'
    test_dir = '/opt/ml/input/data/eval'
    model_dir = '/opt/ml/models'
    submission_dir = '/opt/ml/submissions'
    ensemble_dir = '/opt/ml/ensemble'
    n_fold = 5


class HyperParameter:
    lr = 1e-5
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    NUM_EPOCHS = 5
    train_log_interval = 20  # logging할 iteration의 주기
    patience = 3 # early stopping
