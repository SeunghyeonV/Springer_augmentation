from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

root_dir = 'F:/graphs/imagetest/'
imagename_list = [f for f in listdir(root_dir) if isfile(join(root_dir, f))] #get filename 'XX.png'
print(imagename_list)

#get full dir names in list 'image/directory/XX.png'
filedir_list = [os.path.join(root_dir, imagename_list[i]) for i in range(len(imagename_list))] 
# print(filedir_list)

# 130 images total
fig = plt.figure(figsize=(32, 32))
rows = 13
columns = 10

for i in range(rows*columns):
    car_image = Image.open(filedir_list[i])
    fig.add_subplot(rows, columns, i+1, edgecolor="black")
    plt.imshow(car_image)    
    plt.axis('off')
plt.show()
