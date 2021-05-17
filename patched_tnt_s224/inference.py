import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from include import * 
from configure import * 
from fairseq_model import * 

from patch_dataset import TestBmsDataset, null_collate, null_augment, load_tokenizer, make_fold
from utils import Logger, time_to_str, normalize_inchi, compute_lb_score 

is_mixed_precision = False

import torch.cuda.amp as amp
if is_mixed_precision:
    class AmpNet(Net):
        @torch.cuda.amp.autocast()
        def forward(self, *args):
            return super(AmpNet, self).forward(*args)
else:
    AmpNet = Net

def do_predict(net, tokenizer, test_loader):

    text = []

    start_timer = timer()
    test_num = 0
    for t, batch in enumerate(test_loader):
        batch_size = len(batch['index'])
        # image = batch['image'].cuda()
        patch = batch['patch'].cuda()
        coord = batch['coord'].cuda()
        mask = batch['patch_pad_mask'].cuda()

        net.eval()
        with torch.no_grad():
            #k = net(image)
            k = net.forward_argmax_decode(patch, coord, mask)

            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)
            text.extend(k)

        test_num += batch_size
        print('\r %8d / %d  %s' % (test_num, len(test_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)
         
    assert(test_num == len(test_loader.dataset))
    print('')
    return text    

def run_submit():

    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])

    fold = 3
    out_dir = '/ext_data2/comvis/khanhdtq/bms/tnt_patch16_s0.8/fold%d' % fold
    initial_checkpoint = out_dir + '/checkpoint/00922000_model.pth'

    is_norm_ichi = True

    ## setup  ----------------------------------------
    mode = 'local' #'remote'     

    if mode == 'local':
        submit_dir = out_dir + '/valid/%s-%s-gpu%d'%(mode, 
                                initial_checkpoint[-18:-4], gpu_no)
    elif mode == 'remote':
        submit_dir = out_dir + '/test/%s-%s-gpu%d'%(mode, 
                                initial_checkpoint[-18:-4],gpu_no)

    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('is_norm_ichi = %s\n' % is_norm_ichi)
    log.write('\n')    

    ## dataset ------------------------------------
    tokenizer = load_tokenizer()
    if mode == 'remote':
        df_test = make_fold('test')
    elif mode == 'local':
        df_train, df_test = make_fold('train-%d' % fold)
    
    df_test = df_test.sort_values('length').reset_index(drop=True)
    test_dataset = TestBmsDataset(df_test, tokenizer,
                            mode='valid' if mode == 'local' else 'test')
    test_dataloader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,  
        collate_fn  = lambda batch: null_collate(batch, False),      
    )
    log.write('mode : %s\n'%(mode))
    log.write('test_dataset : \n%s\n'%(test_dataset))

    ## net ----------------------------------------
    tokenizer = load_tokenizer()
    net = AmpNet().cuda()
    net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)
    net = torch.jit.script(net)

    #---
    start_timer = timer()
    predict = do_predict(net, tokenizer, test_loader)
    log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))

    #----
    if is_norm_ichi:
        predict = [normalize_inchi(t) for t in predict] 

    df_submit = pd.DataFrame()
    df_submit.loc[:,'image_id'] = df_test.image_id.values
    df_submit.loc[:,'InChI'] = predict #
    df_submit.to_csv(submit_dir + '/submit.csv', index=False)

    log.write('submit_dir : %s\n' % (submit_dir))
    log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
    log.write('df_submit : %s\n' % str(df_submit.shape))
    log.write('\n')        

    if mode == 'local':
        truth = df_test['InChI'].values.tolist()
        lb_score = compute_lb_score(predict, truth)

        log.write('lb_score  = %f\n'%lb_score.mean())
        log.write('is_norm_ichi = %s\n' % is_norm_ichi)
        log.write('\n')
        
        df_eval = df_submit.copy()
        df_eval.loc[:,'truth']=truth
        df_eval.loc[:,'lb_score']=lb_score
        df_eval.loc[:,'length'] = df_test['length']
        df_eval.to_csv(submit_dir + '/df_eval.csv', index=False)        

if __name__ == '__main__':
    run_submit()
