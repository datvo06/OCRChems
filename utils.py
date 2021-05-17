import math
import time
import torch
import os
import random
import numpy as np
import Levenshtein



# ====================================================
# CFG
# ====================================================

class CFG_vgg16:       # CFG_vgg16 coord
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'vgg16'      # vgg16, efficientnet_b7 maybe?
    enc_size = 512 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 288
    scheduler='CosineAnnealingLR'
    epochs=20
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size= 32
    weight_decay=1e-6
    use_coord=True
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=9
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/vgg16_coord_fold0_best.pth' #prev b1
    pred_model = './saved_model/vgg16_coord_fold0_best.pth'


class CFG_effv2:       # CFG_eff_v2
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'efficientnet_v2s'      # vgg16, efficientnet_b7 maybe?
    enc_size = 1792 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 288
    scheduler='CosineAnnealingLR'
    epochs=20
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size= 32
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=5
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/efficientnet_v2s_fold0_best.pth' #prev b1
    pred_model = './saved_model/efficientnet_v2s_fold0_best.pth'


class CFG_eff_b3_pruned:
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'efficientnet_b3_pruned'      # vgg16, efficientnet_b7 maybe?
    enc_size = 1536 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 320
    scheduler='CosineAnnealingLR'
    epochs=30
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size= 30
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=12
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/efficientnet_b3_pruned_fold0_5.017451189881982.pth' #prev b1
    pred_model = './saved_model/efficientnet_b3_pruned_fold0_best.pth'
    use_coord = False


class CFG_eff_b3:
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'efficientnet_b3'      # vgg16, efficientnet_b7 maybe?
    enc_size = 1536 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 320
    scheduler='CosineAnnealingLR'
    epochs=20
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size=20
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=10
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/efficientnet_b3_fold0_5.702079869977188.pth' #prev b1
    pred_model = './saved_model/efficientnet_b3_fold0_best.pth'
    use_coord=False


class CFG_eff_b1:   # CFG eff b1
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'efficientnet_b1'      # vgg16, efficientnet_b7 maybe?
    enc_size = 1280 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 320
    scheduler='CosineAnnealingLR'
    epochs=20
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size= 30
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=15
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/efficientnet_b1_fold0_3.3411792368188205.pth'
    pred_model = './saved_model/efficientnet_b1_fold0_best.pth'
    use_coord = False


class CFG_eff_b1_old:   # CFG eff b1
    debug =  True
    apex = False
    max_len = 275
    print_freq = 10000
    num_workers=4                       # Prev: 4
    model_name = 'efficientnet_b1'      # vgg16, efficientnet_b7 maybe?
    enc_size = 1280 # 1280 for b1, 1536 for b3
    samp_size = 1000
    size = 288
    scheduler='CosineAnnealingLR'
    epochs=20
    T_max = 4
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size= 30
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5
    seed=42
    n_fold=15
    trn_fold = 0
    train=True
    train_path = "train/"
    test_path = "test/"
    prep_path = 'preprocessed-stuff/'
    prev_model = './saved_model/efficientnet_b1_fold0_3.2892104813739538.pth'
    pred_model = './saved_model/efficientnet_b1_fold0_best.pth'
    use_coord = False


# CFG = CFG_eff_b3_pruned
# CFG = CFG_eff_b1
# CFG = CFG_eff_b3
CFG = CFG_eff_b1_old


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def init_logger(log_file='inference.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False      # True


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


if not os.path.exists(CFG.prep_path):
    os.makedirs(CFG.prep_path)
