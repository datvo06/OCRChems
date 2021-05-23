from pathlib import Path
from tokenizer import split_form, split_form2, tokenizer
import pandas as pd
import os


PROJECT_DIR = Path('.')
INPUT_DIR = PROJECT_DIR
EXTRA_APPROVED_LABELS_PATH = INPUT_DIR / 'extra_approved_InChIs.csv'

if not os.path.exists('extra_gen.csv'):
    out_file = open('extra_gen.csv', 'w')
    out_file.write("index,file_path,InChI_length,InChI_text\n")
    # Open the oriignal extra approved
    EXTRA_APPROVED_LABELS = pd.read_csv(EXTRA_APPROVED_LABELS_PATH)
    for i, row in EXTRA_APPROVED_LABELS.iterrows():
        fp = os.path.join('extra_gen', str(i) + ".png")
        if not os.path.exists(fp):
            continue
        inchi = row['InChI']
        inchi_1 = inchi.split("/")[1]
        inchi_text = split_form(inchi_1) + ' ' +  split_form2(
            '/'.join(inchi.split('/')[2:]))
        seq = tokenizer.text_to_sequence(inchi_text)
        length = len(seq) - 2       # removed sos and eos
        out_file.write(f"{i},\"{fp}\",{length},\"{inchi_text}\"\n")
    out_file.close()
