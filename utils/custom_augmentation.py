import random
import numpy as np
from PIL import Image, ImageOps
import torch

class custom_augmentation_pixel(object):
    def __init__(self, p=0.1, default_score=0, score=0):
        self.p = p
        self.default_score = default_score
        self.score = score

    def __call__(self, img):
        pix_val = list(img.getdata())
        new_pix_val = [[] for _ in range(img.size[0])] #[[] for _ in range(32)]
        empty_image = Image.new('RGB', (img.size[0], img.size[1])) #append pixels to empty image
        k = 0
        while k < len(pix_val):
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    new_pix_val[i].append(pix_val[j+k]) #append len(first_row_of_an_image) pixels to nth row
                    if torch.rand(1) < self.p:
                        # new_pix_val[i][j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        new_pix_val[i][j] = pix_val[i][j]
                k += len(new_pix_val)

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                empty_image.putpixel((i, j), new_pix_val[j][i])

        return empty_image 

    def __repr__(self):
        return "custom augmentation"
