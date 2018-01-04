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


def SelectiveSearch():
    global candidates, x, y, w, h
    global img
    # loading astronaut image
    # img = skimage.data.astronaut()
    img = Image.open("dog0.jpg")
    img = np.array(img)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    
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

    global fix, ax

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    img = Image.open("dog0.jpg")
    i=0
    image_size = 224
    img_set = np.zeros((1, image_size, image_size, 3))
    for x, y, w, h in candidates:
        # print(x, y, w, h)
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        img_cut = img.crop((x,y,x+w,y+h))
        img_cut = WarpedRegion(image_size, img_cut)
        img_cut = np.array(img_cut)
        # plt.imshow(img_cut)
        # plt.show()
        img_cut = np.expand_dims(np.array(img_cut, dtype=np.float32), axis=0)
        img_set = np.concatenate((img_set, img_cut),axis=0)
        i=i+1
    # plt.show()
    # 剪取圖片之集合
    img_set = img_set[1:, :, :]
    model = ResNet50(weights='imagenet')
    img_set = preprocess_input(img_set)
    predict_arr = []
    i=0
    for x in img_set:
        y = np.expand_dims(x, axis=0)
        features = model.predict([y])
        # print('Predicted:', decode_predictions(features, top=1)[0])
        predict_arr.append(decode_predictions(features, top=1)[0])
        i=i+1
    pprint.pprint(predict_arr)
    return img_set

def WarpedRegion(size, img):
    img = img.resize((size, size), Image.BILINEAR)
    return img

def ResNet50Predict(img_set):
    model = ResNet50(weights='imagenet')
    img_set = preprocess_input(img_set)
    predict_arr = []
    i=0
    for x in img_set:
        y = np.expand_dims(x, axis=0)
        features = model.predict([y])
        # print('Predicted:', decode_predictions(features, top=1)[0])
        predict_arr.append(decode_predictions(features, top=1)[0])
        i=i+1
    pprint.pprint(predict_arr)
    return predict_arr
if __name__ == "__main__":
    img_set = SelectiveSearch()
    print(len(img_set))
    print(img_set.shape)
    # ResNet50Predict(img_set)
