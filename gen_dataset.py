from pathlib import Path
from io import BytesIO
import re
import copy
import logging
import os

import numpy as np
import pandas as pd

import lxml.etree as et
import cssutils

from PIL import Image
import cairosvg
import sys
from skimage.transform import resize

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawOptions


PROJECT_DIR = Path('.')
INPUT_DIR = PROJECT_DIR
TRAIN_DATA_PATH = INPUT_DIR / 'train'
TRAIN_LABELS_PATH = INPUT_DIR / 'train_labels.csv'
EXTRA_APPROVED_LABELS_PATH = INPUT_DIR / 'extra_approved_InChIs.csv'
EXTRA_APPROVED_LABELS = pd.read_csv(EXTRA_APPROVED_LABELS_PATH)

cssutils.log.setLevel(logging.CRITICAL)

np.set_printoptions(edgeitems=30, linewidth=180)
print('RDKit version:', rdkit.__version__)
# Use a specific version of RDKit with known characteristics so that we can reliably manipulate output SVG.

def one_in(n):
    return np.random.randint(n) == 0 and True or False


def yesno():
    return one_in(2)


def svg_to_image(svg, convert_to_greyscale=True):
    svg_str = et.tostring(svg)
    # TODO: would prefer to convert SVG dirrectly to a numpy array.
    png = cairosvg.svg2png(bytestring=svg_str)
    image = np.array(Image.open(BytesIO(png)), dtype=np.float32)
    # Naive greyscale conversion.
    if convert_to_greyscale:
        image = image.mean(axis=-1)
    return image


def elemstr(elem):
    return ', '.join([item[0] + ': ' + item[1] for item in elem.items()])


# Streches the value range of an image to be exactly 0, 1, unless the image appears to be blank.
def stretch_image(img, blank_threshold=1e-2):
    img_min = img.min()
    img = img - img_min
    img_max = img.max()
    if img_max < blank_threshold:
        # seems to be blank or close to it
        return img
    img_max = img.max()
    if img_max < 1.0:
        img = img/img_max
    return img


def random_molecule_image(inchi, drop_bonds=True, add_noise=True, render_size=1200, margin_fraction=0.2):
    # Note that the original image is returned as two layers: one for atoms and one for bonds.
    #mol = Chem.MolFromSmiles(smiles)
    mol = Chem.inchi.MolFromInchi(inchi)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(render_size, render_size)
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.additionalAtomLabelPadding = np.random.uniform(0, 0.3)
    options.bondLineWidth = int(np.random.uniform(1, 4))
    options.multipleBondOffset = np.random.uniform(0.05, 0.2)
    options.rotate = np.random.uniform(0, 360)
    options.fixedScale = np.random.uniform(0.05, 0.07)
    options.minFontSize = 20
    options.maxFontSize = options.minFontSize + int(np.round(np.random.uniform(0, 36)))
    d.SetFontSize(100)
    d.SetDrawOptions(options)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg_str = d.GetDrawingText()
    # Do some SVG manipulation
    svg = et.fromstring(svg_str.encode('iso-8859-1'))
    atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    bond_elems = svg.xpath(r'//svg:path[starts-with(@class,"bond-")]', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    # Change the font.
    font_family = np.random.choice([
        'serif',
        'sans-serif'
    ])
    for elem in atom_elems:
        style = elem.attrib['style']
        css = cssutils.parseStyle(style)
        css.setProperty('font-family', font_family)
        css_str = css.cssText.replace('\n', ' ')
        elem.attrib['style'] = css_str
    # Create the original image layers.
    # TODO: separate atom and bond layers
    bond_svg = copy.deepcopy(svg)
    # remove atoms from bond_svg
    for elem in bond_svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_bond_img = svg_to_image(bond_svg)
    atom_svg = copy.deepcopy(svg)
    # remove bonds from atom_svg
    for elem in atom_svg.xpath(r'//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_atom_img = svg_to_image(atom_svg)
    if drop_bonds:
        num_bond_elems = len(bond_elems)
        if one_in(3):
            while True:
                # drop a bond
                # Let's leave at least one bond!
                if num_bond_elems > 1:
                    bond_elem_idx = np.random.randint(num_bond_elems)
                    bond_elem = bond_elems[bond_elem_idx]
                    bond_parent_elem = bond_elem.getparent()
                    if bond_parent_elem is not None:
                        bond_parent_elem.remove(bond_elem)
                        num_bond_elems -= 1
                else:
                    break
                if not one_in(4):
                    break
    img = svg_to_image(svg) > 254
    img = 1*img  # bool → int
    # Calculate the margins.
    black_indices = np.where(img == 0)
    row_indices, col_indices = black_indices
    if len(row_indices) >= 2:
        min_y, max_y = row_indices.min(), row_indices.max() + 1
    else:
        min_y, max_y = 0, render_size
    if len(col_indices) >= 2:
        min_x, max_x = col_indices.min(), col_indices.max() + 1
    else:
        min_x, max_x = 0, render_size
    margin_size = int(np.random.uniform(0.8*margin_fraction, 1.2*margin_fraction)*max(max_y - min_y, max_x - min_x))
    min_y, max_y = max(min_y - margin_size, 0), min(max_y + margin_size, render_size)
    min_x, max_x = max(min_x - margin_size, 0), min(max_x + margin_size, render_size)
    img = img[min_y:max_y, min_x:max_x]
    img = img.reshape([img.shape[0], img.shape[1]]).astype(np.float32)
    orig_bond_img = orig_bond_img[min_y:max_y, min_x:max_x]
    orig_atom_img = orig_atom_img[min_y:max_y, min_x:max_x]
    scale = np.random.uniform(0.2, 0.4)
    sz = (np.array(orig_bond_img.shape[:2], dtype=np.float32)*scale).astype(np.int32)
    orig_bond_img = resize(orig_bond_img, sz, anti_aliasing=True)
    orig_atom_img = resize(orig_atom_img, sz, anti_aliasing=True)
    img = resize(img, sz, anti_aliasing=False)
    img = img > 0.5
    if add_noise:
        # Add "salt and pepper" noise.
        salt_amount = np.random.uniform(0, 0.3)
        salt = np.random.uniform(0, 1, img.shape) < salt_amount
        img = np.logical_or(img, salt)
        pepper_amount = np.random.uniform(0, 0.001)
        pepper = np.random.uniform(0, 1, img.shape) < pepper_amount
        img = np.logical_or(1 - img, pepper)

    img = img.astype(np.uint8)  # boolean -> uint8
    orig_bond_img = 1 - orig_bond_img/255
    orig_atom_img = 1 - orig_atom_img/255
    # Stretch the range of the atom and bond images so that the min is 0 and the max. is 1
    orig_bond_img = stretch_image(orig_bond_img)
    orig_atom_img = stretch_image(orig_atom_img)
    return img, orig_bond_img, orig_atom_img


def get_random_molecule_image(imol):
    inchi = EXTRA_APPROVED_LABELS['InChI'][imol]
    img, orig_bond_img, orig_atom_img = random_molecule_image(inchi)
    a = (255*(1 - img)).astype(np.uint8)
    img_pil = Image.fromarray(a).convert('L')
    return img_pil


if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    save_dir = "extra_gen"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(start, end):
        if os.path.exists(os.path.join(save_dir, str(i) + ".png")):
            continue
        try:
          img_pil = get_random_molecule_image(i)
          img_pil.save(os.path.join(save_dir, str(i) + ".png"))
        except:
          print("error, cant write file \n")
          continue
