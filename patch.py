import numpy as np
import io
from utils import CFG_tnt as CFG
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import os
from utils import write_pickle_to_file

# tools to make patches ...
# see https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-inference/data
def remove_rotate(image, orientation):
    l = orientation
    if l == 1:
        image = np.rot90(image,-1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)
    return image


def resize_image(image, scale=1):
    if scale==1 :
        f = CFG.pixel_scale * 58/36  #1.2414 #0.80555
        b = int(round(36*0.5))

    if scale==2 :
        f = CFG.pixel_scale * 1
        b = int(round(58*0.5))

    image = image[b:-b,b:-b] #remove border
    if not np.isclose(1,f, rtol=1e-02, atol=1e-02):
        h, w = image.shape
        fw = int(round(f*w))
        fh = int(round(f*h))
        image = cv2.resize(image, dsize=(fw, fh), interpolation=cv2.INTER_AREA)
    return image


def repad_image(image, multiplier=16):
    h, w = image.shape
    fh = int(np.ceil(h/multiplier))*multiplier
    fw = int(np.ceil(w/multiplier))*multiplier
    m  = np.full((fh, fw), 255, np.uint8)
    m[0:h, 0:w] = image
    return m


def image_to_patch(image, patch_size, pixel_pad, threshold=0):
    p = pixel_pad
    h, w = image.shape

    x, y = np.meshgrid(np.arange(w // patch_size), np.arange(h // patch_size))
    yx = np.stack([y, x], 2).reshape(-1, 2)

    s = patch_size + 2*p
    m = torch.from_numpy(image).reshape(1, 1, h, w).float()
    k = F.unfold(m, kernel_size=s, stride=patch_size, padding=p)
    k = k.permute(0, 2, 1).reshape(-1, s * s)
    k = k.data.cpu().numpy().reshape(-1, s, s)
    #print(k.shape)

    sum = (1 - k[:, p:-p, p:-p]/255).reshape(len(k), -1).sum(-1)
    i = np.where(sum > threshold)
    #print(sum)
    patch = k[i]
    coord = yx[i]
    return  patch, coord


def patch_to_image(patch, coord, width, height):
    image = np.full((height,width), 255, np.uint8)
    p = CFG.pixel_pad
    patch = patch[:, p:-p, p:-p]
    num_patch = len(patch)

    for i in range(num_patch):
        y,x = coord[i]
        x = x * CFG.patch_size
        y = y * CFG.patch_size
        image[y:y+CFG.patch_size,x:x+CFG.patch_size] = CFG.patch[i]
        cv2.rectangle(image, (x, y), (x + CFG.patch_size,
                                      y + CFG.patch_size),
                      128, 1)
    return image

#<todo>
# np compression is very slow!!!
# https://stackoverflow.com/questions/39035983/compress-zip-numpy-arrays-in-memory
def compress_array(k):
    compressed_k = io.BytesIO()
    np.savez_compressed(compressed_k, k)
    return compressed_k

def uncompress_array(compressed_k):
    compressed_k.seek(0)
    k  = np.load(compressed_k,allow_pickle=True)['arr_0']
    return k

#<todo> add token to mark 4 corner of image
#############################################################################################
def make_chessbord_image(w, h, patch_size=16):
    m = np.zeros((h, w), np.float32)
    s = patch_size
    u = 1
    for y in range(0, h, s):
        v = u
        for x in range(0, w, s):
            m[y:y + s, x:x + s] = v
            v *= -1
        u *= -1
    m = ((0.5 * m + 0.5) * 255).astype(np.uint8)
    return m


def run_make_patch_data(mode='train'):
    if mode == 'train':
        df = pd.read_csv(CFG.data_dir+'/df_train_image_meta.csv')
        folder = 'train_patch16_s%0.3f'%(CFG.pixel_scale)

    if mode == 'test':
        df = pd.read_csv(CFG.data_dir+'/df_test_image_meta.csv')
        folder = 'test_patch16_s%0.3f'%(CFG.pixel_scale)

    #---
    # dump_dir = '/ext_data2/comvis/khanhdtq/bms/test_patch16_s0.800'
    dump_dir = f'{CFG.patch_dump_dir}/{CFG.mode}_patch16_s{CFG.pixel_scale:0.3f}'
    os.makedirs(dump_dir, exist_ok=True)
    e = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    for f in e:
        for g in e:
            for h in e:
                os.makedirs(dump_dir+'/%s/%s/%s/%s'%(folder,f,g,h),exist_ok=True)


    #---
    num_patch = []
    print('Number of image to process:', len(df))
    for i,d in df.iterrows():
        if i%1000==0: print(i, d.image_id)
        image_id = d.image_id
        scale = d.scale
        orientation = d.orientation

        image_file = CFG.data_dir + '/%s/%s/%s/%s/%s.png' % (mode, image_id[0], image_id[1], image_id[2], image_id)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        if mode=='test':
            image = remove_rotate(image, orientation)

        image = resize_image(image, scale)
        image = repad_image(image, CFG.patch_size)  # remove border and repad
        # print(image.shape)

        k, yx = image_to_patch(image, CFG.patch_size,
                               CFG.pixel_pad, threshold=4)
        # k, yx = image_to_patch(image, patch_size, pixel_pad, threshold=4)
        if CFG.debug:  #debug
            for y, x in yx:
                x = x * CFG.patch_size
                y = y * CFG.patch_size
                cv2.rectangle(image, (x, y),
                              (x + CFG.patch_size,
                               y + CFG.patch_size), 128, 1)

            # image_show('image', image, resize=1)
            cv2.waitKey(0)

        #-------------------------------------------
        h,w = image.shape
        yx  = yx.astype(np.int32)
        k   = compress_array(k.astype(np.uint8))

        write_pickle_to_file(dump_dir + '/%s/%s/%s/%s/%s.pickle' % (
            folder, d.image_id[0], d.image_id[1], d.image_id[2], d.image_id),
            {'patch':k, 'coord': yx, 'width': w,'height': h}
        )
        num_patch.append(len(yx))


    df_patch = pd.DataFrame({
        'image_id': df.image_id.values,
        'num_patch': num_patch,
    })
    # df_patch.to_csv(data_dir+'/df_test_patch_s%0.3f.csv'%(pixel_scale), index=False)
    df_patch.to_csv(CFG.data_dir + f'/df_{mode}_patch_{CFG.pixel_scale:0.3f}.csv', index=False)
    exit(0)

if __name__ == '__main__':
    run_make_patch_data(mode='train')
    run_make_patch_data(mode='test')
