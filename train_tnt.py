import os
import Levenshtein
import torch
import torch.nn.functional as F
import numpy as np
import timer
from utils import CFG_tnt as CFG, time_to_str, LOGGER
from model_tnt import seq_cross_entropy_loss, seq_focal_cross_entropy_loss
from tokenizer import tokenizer, train
from patch_dataset import BmsDataset, make_fold, DataLoader, null_collate
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn.parallel.data_parallel import data_parallel
from model_tnt import TnTNet
from optim_tnt import Lookahead, RAdam, get_learning_rate
import torch.cuda.amp as amp



def get_train_file_path(image_id):
    return CFG.train_path + '{}/{}/{}/{}.png'.format(
        image_id[0], image_id[1], image_id[2], image_id
    )



def np_loss_cross_entropy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p,1e-6,1))
    loss = loss.mean()
    return loss



def do_valid(net, tokenizer, valid_loader):
    valid_probability = []
    valid_truth = []
    valid_length = []
    valid_num = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        length = batch['length']
        token  = batch['token' ].cuda()
        token_pad_mask = batch['token_pad_mask' ].cuda()
        #image  = batch['image' ].cuda()
        num_patch = batch['num_patch']
        patch  = batch['patch' ].cuda()
        coord  = batch['coord' ].cuda()
        patch_pad_mask  = batch['patch_pad_mask' ].cuda()

        with torch.no_grad():
            #logit = data_parallel(net, (patch, coord, token, patch_pad_mask, token_pad_mask)) #net(image, token, length)
            logit = net(patch, coord, token, patch_pad_mask, token_pad_mask)
            probability = F.softmax(logit,-1)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(token.data.cpu().numpy())
        valid_length.extend(length)
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.sampler),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    assert(valid_num == len(valid_loader.sampler)) #len(valid_loader.dataset))
    #print('')
    #----------------------
    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)
    truth   = np.concatenate(valid_truth)
    length  = valid_length


    #----
    p = probability[:,:-1].reshape(-1, len(tokenizer.stoi.keys()))
    t = truth[:,1:].reshape(-1)

    non_pad = np.where(t!=tokenizer.stoi['<pad>'])[0] #& (t!=STOI['<sos>'])
    p = p[non_pad]
    t = t[non_pad]
    loss = np_loss_cross_entropy(p, t)

    #----
    lb_score = 0
    if 1:
        score = []
        for i,(p, t) in enumerate(zip(predict, truth)):
            t = truth[i][1:length[i]-1]
            p = predict[i][:length[i]-2]
            t = tokenizer.one_predict_to_inchi(t)
            p = tokenizer.one_predict_to_inchi(p)
            s = Levenshtein.distance(p, t)
            score.append(s)
        lb_score = np.mean(score)

    return [loss, lb_score]


def run_train():
    fold = CFG.fold
    out_dir = f'{CFG.out_dir}{CFG.pixel_scale:.3f}/fold%d' % CFG.fold
    os.makedirs(out_dir, exist_ok=True)
    initial_checkpoint = None

    debug = CFG.debug
    start_lr = CFG.encoder_lr #0.00005# 1
    batch_size = CFG.batch_size # 24


    ## setup  ----------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    LOGGER.info('\tout_dir  = %s\n' % out_dir)
    LOGGER.info('\n')

    ## dataset ------------------------------------

    df_train, df_valid = make_fold('train-%d' % fold)
    df_valid = df_valid.iloc[:5_000]

    train_dataset = BmsDataset(df_train,tokenizer)
    valid_dataset = BmsDataset(df_valid,tokenizer)

    train_loader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        #sampler=UniformLengthSampler(train_dataset, is_shuffle=True), #200_000
        batch_size=batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        #sampler=UniformLengthSampler(valid_dataset, 5_000),
        sampler=SequentialSampler(valid_dataset),
        batch_size=24,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=null_collate,
    )

    LOGGER.info('train_dataset : \n%s\n' % (train_dataset))
    LOGGER.info('valid_dataset : \n%s\n' % (valid_dataset))
    LOGGER.info('\n')

    ## net ----------------------------------------
    LOGGER.info('** net setting **\n')
    if CFG.use_mixed:
        scaler = amp.GradScaler()
    net = TnTNet().cuda()  #  AmpNet().cuda()


    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch     = f['epoch']
        state_dict = f['state_dict']

        #---
        # state_dict = {k.replace('cnn.e.','cnn.'):v for k,v in state_dict.items()}
        # del state_dict['text_pos.pos']
        # del state_dict['cnn.head.weight']
        # del state_dict['cnn.head.bias']
        # net.load_state_dict(state_dict, strict=False)

        #---
        net.load_state_dict(state_dict, strict=True)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    LOGGER.info('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    LOGGER.info('\n')

    # -----------------------------------------------
    if 0:  ##freeze
        for p in net.encoder.parameters(): p.requires_grad = False

    optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr), alpha=0.5, k=5)
    # optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)

    num_iteration = 80000 * 1000
    iter_log = 1000
    iter_valid = 1000
    iter_save = list(range(0, num_iteration, 1000))  # 1*1000

    LOGGER.info('optimizer\n  %s\n' % (optimizer))
    LOGGER.info('\n')

    ## start training here! ##############################################
    LOGGER.info('** start training here! **\n')
    LOGGER.info('   is_mixed_precision = %s \n' % str(CFG.use_mixed))
    LOGGER.info('   batch_size = %d\n' % (batch_size))
    LOGGER.info('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    LOGGER.info('                      |----- VALID ---|---- TRAIN/BATCH --------------\n')
    LOGGER.info('rate     iter   epoch | loss  lb(lev) | loss0  loss1  | time          \n')
    LOGGER.info('----------------------------------------------------------------------\n')
             # 0.00000   0.00* 0.00  | 0.000  0.000  | 0.000  0.000  |  0 hr 00 min

    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
            '%4.3f  %5.2f  | ' % (*valid_loss,) + \
            '%4.3f  %4.3f  %4.3f  | ' % (*loss,) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))

        return text

    # ----
    valid_loss = np.zeros(2, np.float32)
    train_loss = np.zeros(3, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss0 = torch.FloatTensor([0]).cuda().sum()
    loss1 = torch.FloatTensor([0]).cuda().sum()
    loss2 = torch.FloatTensor([0]).cuda().sum()

    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    while iteration < num_iteration:

        for t, batch in enumerate(train_loader):

            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass

            if (iteration % iter_valid == 0):
                #if iteration != start_iteration:
                    valid_loss = do_valid(net, tokenizer, valid_loader)  #
                    pass

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                LOGGER.info(message(mode='log') + '\n')

            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            length = batch['length']
            token  = batch['token' ].cuda()
            token_pad_mask = batch['token_pad_mask' ].cuda()
            #image  = batch['image' ].cuda()
            num_patch = batch['num_patch']
            patch  = batch['patch' ].cuda()
            coord  = batch['coord' ].cuda()
            patch_pad_mask = batch['patch_pad_mask' ].cuda()


            # ----
            net.train()
            optimizer.zero_grad()

            if CFG.use_mixed:
                with amp.autocast():
                    #assert(False)
                    #logit = data_parallel(net, (patch, coord, token, patch_pad_mask, token_pad_mask)) #net(image, token, length)
                    logit = net(patch, coord, token, patch_pad_mask, token_pad_mask)
                    loss0 = seq_cross_entropy_loss(logit, token, length)
                    #loss0 = seq_anti_focal_cross_entropy_loss(logit, token, length)

                scaler.scale(loss0).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()


            else:
                #logit = net(patch, coord, token, patch_pad_mask, token_pad_mask)
                logit = data_parallel(net, (patch, coord, token, patch_pad_mask, token_pad_mask))
                loss0 = seq_cross_entropy_loss(logit, token, length)
                (loss0).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                optimizer.step()

            # print statistics  --------
            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)

            # debug--------------------------
            if debug:
                pass

    LOGGER.info('\n')


# main #################################################################
if __name__ == '__main__':
    run_train()
