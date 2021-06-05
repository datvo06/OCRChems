import pandas as pd
import sys

p_src = sys.argv[1]
p_tgt = sys.argv[2]
src = pd.read_csv(p_src)
img_id_list = src['image_id'].to_list()
map_to_key = dict(zip(img_id_list, list(range(len(img_id_list)))))

tgt = pd.read_csv(p_tgt)
tgt['idx'] = tgt['image_id'].map(map_to_key)
tgt.sort_values(['idx'], ascending=True, inplace=True)
tgt.drop('idx', 1, inplace=True)
tgt.to_csv('sorted.csv')
