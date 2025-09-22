import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image


MULTIPLIER = 3

def transforming(image,mask):
    t = A.Compose([
    A.HorizontalFlip(p=0.5),  # 50% chance to flip the image horizontally
    A.Rotate(limit=15, p=0.5),  # 50% chance to rotate the image within the range [-15, 15] degrees
    #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)  # 50% chance to adjust brightness and contrast
    ])
    return t(image=image,mask = mask) 

def check(pair):
    im = Image.fromarray(pair["image"])
    im.save("image.jpeg")
    mask = pair["mask"]
    mask.save("mask.jpeg")
    

def augumenting(images,masks):
    new_images = images.copy()
    new_masks = masks.copy()
    
    for i in range(len(images)):
        transformed = transforming(image=images[i,:,:],mask=masks[i,:,:])
        if i == 0:
            check(transformed)
        break
        new_images[i,:,:] = transformed['image']
        new_masks[i,:,:] = transformed['mask']
    # Storing both the versions
    images = np.concatenate((images,  new_images))
    masks = np.concatenate((masks, new_masks))
    return images,masks




