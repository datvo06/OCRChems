import numpy as np
from skimage import color
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import numpy as np
import cv2

inchi = 'InChI=1S/C15H14FN3O3/c1-3-22-15(21)13-12-7-18(2)14(20)10-6-9(16)4-5-11(10)19(12)8-17-13/h4-6,8H,3,7H2,1-2H3'
mol = Chem.MolFromInchi(inchi)
im = Draw.MolToImage(mol, size=(384, 384))
np_im = np.array(im)
np_im = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
_, np_im = cv2.threshold(np_im, 240, 255, cv2.THRESH_BINARY_INV)


def salt_pepper(bin_np_im):
    threshold = 0.01 * 255
    salt_pepper = np.random.rand(bin_np_im.shape[0], bin_np_im.shape[1]) * 255
    bin_np_im[salt_pepper > (255 - threshold)] = 255
    bin_np_im[salt_pepper < threshold] = 0
    bin_np_im = 255 - bin_np_im
    return Image.fromarray(bin_np_im.astype('uint8')).convert('L')

im = salt_pepper(np_im)
im.save('sample.png')
