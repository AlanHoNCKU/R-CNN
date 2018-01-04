# -*- coding: utf-8 -*-
'''
author: Chia Yu
2017/01/02
Region Conventional Neural Network implement
'''

from __future__ import (
    division,
    print_function,
)
import pprint
from skimage import io, data
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image # 剪取圖片
import selectivesearch
import numpy as np
# pretrain model (ResNet50)
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def R_CNN():
    img = Image.open("dog0.jpg")
    img_sel = np.array(img)
    img_lbl, regions = selectivesearch.selective_search(img_sel, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img_sel)
    image_size = 224
    img_set = np.zeros((1, image_size, image_size, 3))
    for x, y, w, h in candidates:
        # print(x, y, w, h)
        # rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        # ax.add_patch(rect)
        img_cut = img.crop((x,y,x+w,y+h))
        img_cut = WarpedRegion(image_size, img_cut)
        img_cut = np.array(img_cut)
        img_cut_preprocess = np.expand_dims(np.array(img_cut, dtype=np.float32), axis=0)
        # 增加一個維度
        model = ResNet50(weights='imagenet')
        img_cut_preprocess = preprocess_input(img_cut_preprocess)
        features = model.predict([img_cut_preprocess])
        print('Predicted:', decode_predictions(features, top=3)[0])
        fig = plt.figure()
        fig.suptitle(decode_predictions(features, top=1)[0], fontsize=14, fontweight='bold') 
        plt.imshow(img_cut)
        plt.show()


def WarpedRegion(size, img):
    img = img.resize((size, size), Image.BILINEAR)
    return img


if __name__ == "__main__":
    R_CNN()