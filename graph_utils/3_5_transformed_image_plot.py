import matplotlib.pyplot as plt
from torchvision import transforms
import sys
sys.path.append("C:/Users/seacl/Desktop/CIFAR100_reconstruction/")
from utils.randaugment import *
import torch
from PIL import Image


img = Image.open('C:/Users/seacl/Desktop/Thesis_research/reinforcement_augmentation/Imagenet/train/hen/n01514859_364.jpeg')
img = img.resize((224,224))

transform = transforms.Compose([CIFAR10Policy_equalize(magnitude=3), transforms.ToTensor()])

transforms_list = [ transforms.Compose([transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_autocontrast(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_brightness(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_color(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_contrast(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_equalize(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_invert(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_posterize(magnitude=9), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_rotate(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_sharpness(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_shearX(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_shearY(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_solarize(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_translateX(magnitude=5), transforms.ToTensor()]),
                    transforms.Compose([CIFAR10Policy_translateY(magnitude=5), transforms.ToTensor()]),
                    ]


    
labels = ['Original', 'Autocontrast', 'Brightness', 'Color', 'Contrast', 'Equalize',
          'Invert', 'Posterize', 'Rotate', 'Sharpness', 'ShearX', 'ShearY',
          'Solarize', 'TranslateX', 'TranslateY']


columns = 3
rows = 5
fig = plt.figure(figsize=(5, 8))

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(transforms_list[i-1](img).permute(1, 2, 0).cpu())
    plt.title(labels[i-1])
    plt.axis('off')
# plt.savefig("C:/Users/seacl/Desktop/figure_aug.png", dpi=200, bbox_inches = 'tight')
plt.show()
