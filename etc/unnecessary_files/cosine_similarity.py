import torch
from torchvision import transforms, datasets, models
import numpy as np
from numpy import dot
from numpy.linalg import norm

#used to compute image cosine similarity
def cosine_similarity(img1, img2):
    toTensor = transforms.ToTensor()
    img1 = toTensor(img1)
    img2 = toTensor(img2)
    flattened_img1 = torch.reshape(img1, (-1,))
    flattened_img2 = torch.reshape(img2, (-1,))
    return dot(flattened_img1, flattened_img2) / (norm(flattened_img1) * norm(flattened_img2))
