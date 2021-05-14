from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pathlib import Path
import sys

# Note: activate rdkit env before running this
# my-rdkit-env

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        return (inchi, False) if (mol is None) else (Chem.MolToInchi(mol), True)
    except: return (inchi, False)


# Segfault in rdkit taken care of, run it with:
# while [ 1 ]; do python normalize_inchis.py && break; done
if __name__=='__main__':
    # Input & Output
    orig_paths = sys.argv[1:]
    norm_path = Path('submission_multiple_norm.csv')

    # Do the job
    N = norm_path.read_text(encoding='UTF-8', errors='ignore').count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')
    print(orig_paths)

    rs = [open(str(orig_path), 'r', errors='ignore') for orig_path in orig_paths]
    w = open(str(norm_path), 'a', buffering=1)

    for _ in range(N):
        for r in rs:
            r.readline()
    for r in rs:
        try:
            line = r.readline()  # this line is the header or is where it segfaulted last time
        except:
            print('error on this line: ')
            print(line)
            print(' ')
    w.write(line)

    for line0 in tqdm(rs[0]):
        lines = [line0]
        for r in rs[1:]:
            lines.append(r.readline())
        splits0 = line0[:-1].split(',')
        inchi0 = ','.join(splits0[1:]).replace('"','')
        image_id = splits0[0]
        for line in lines:
            inchi = ','.join(line[:-1].split(',')[1:]).replace('"','')
            inchi_norm, valid = normalize_inchi(inchi)
            if valid:
                w.write(f'{image_id},"{inchi_norm}"\n')
                break
        if not valid:
            w.write(f'{image_id},"{inchi0}"\n')

    r.close()
    w.close()
