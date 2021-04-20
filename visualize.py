from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A

test = pd.read_csv('sample_submission.csv')


def get_test_file_path(image_id):
    return "test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)


test['file_path'] = test['image_id'].apply(get_test_file_path)

print('test.shape: {}'.format(test.shape))
print(test.head())

plt.figure(figsize=(20, 20))
for i in range(20):
    image = cv2.imread(test.loc[i, 'file_path'])
    plt.subplot(5, 4, i + 1)
    plt.imshow(image)
plt.show()

transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

plt.figure(figsize=(20, 20))

# Fix vertical image for horizontal compound images
for i in range(20):
    image = cv2.imread(test.loc[i, 'file_path'])
    h, w, _ = image.shape
    if h > w:
        image = transform(image=image)['image']
    plt.subplot(5, 4, i+1)
    plt.imshow(image)

plt.show()
