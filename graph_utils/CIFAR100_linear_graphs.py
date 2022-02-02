import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as font_manager

# all font
csfont = {'fontname':'Times New Roman'}
# legend font
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

## Draw linear graphs for each augmentation that shows accuracy change per magnitude

original_MA = [60.39, 60.39, 60.39, 60.39, 60.4]
autocontrast_MA = [62.83, 62.8, 62.87, 62.87, 62.87]
brightness_MA = [70.07, 63.47, 52.63, 30.67, 16.23]
color_MA = [70.03, 68.3, 65.53, 60.53, 51.83]
contrast_MA = [69.97, 66.7, 59.9, 47.1, 21.5]
equalize_MA = [30.57, 30.56, 30.57, 30.58, 30.56]
invert_MA = [12.2, 12.17, 12.17, 12.17, 12.17]
posterize_MA =  [70.4, 69.97, 63.87, 23.2, 4.63]
rotate_MA = [68.93, 67.97, 63.97, 50.7, 36.77]
sharpness_MA = [69.73, 68.9, 67.8, 65.03, 61.33]
shearX_MA = [68.87, 68.9, 66.57, 63.4, 58.23]
shearY_MA = [69.97, 68, 66.43 ,62.63 ,58.63]
solarize_MA = [53.17, 39.93, 25.6, 9.53, 12.17]
translateX_MA = [69.33, 66.2, 61, 54.83, 35.63]
translateY_MA = [69.7, 64.67, 54.77, 42.9, 21.83]

original_AA = [43.24, 43.24, 43.24, 43.24, 43.25]
autocontrast_AA = [30.07, 21.43, 26.4, 24.07, 31.27]
brightness_AA = [33.43, 17.93, 12.9, 7.67, 5.73]
color_AA = [35.77, 23.57, 26.7, 22.2, 23.8]
contrast_AA = [32.97, 19.93, 14.97, 7.93, 6.7]
equalize_AA = [9.43, 7.5, 7.3, 8.83, 9.5]
invert_AA = [3, 4.17, 4.5, 3.5, 2.7]
posterize_AA = [26.43, 26.97, 28.97, 10, 2.73]
rotate_AA = [26.23, 19.37, 16.5, 18.23, 14.4]
sharpness_AA = [28.23, 25, 26.57, 31.07, 24.33]
shearX_AA = [27.87, 23.6, 25.4, 30.9, 24.1]
shearY_AA = [28.33, 23.17, 24.73, 30, 22.83]
solarize_AA = [12.77, 5.33, 5.47, 2.6, 3.8]
translateX_AA = [27.6, 24.1, 23.5, 25.97, 13.87]
translateY_AA = [27.83, 22.87, 20.1, 19.23, 8]


Magnitude = [1,3,5,7,9]
plt.xticks(Magnitude, Magnitude)
plt.ylim(0,80)

plt.title("CIFAR-100 Model accuracy range", **csfont, fontsize=14)
plt.xlabel("Magnitude", **csfont)
plt.ylabel("Accuracy", **csfont)
x_axis = range(0, 15)
# cmap = plt.get_cmap('tab20c') #automatically adjust color by applying color=cmap(i)

# highest value and coordinates
original_MA_max = max(original_MA)
original_MA_max_index = x_axis[original_MA.index(original_MA_max)]
original_MA_mx_coord = 8
original_MA_my_coord = original_MA_max+10
plt.annotate("[O]{}%".format(original_MA_max), xy=(original_MA_max_index, original_MA_max), xytext=(original_MA_mx_coord,original_MA_my_coord), **csfont)

# highest value and coordinates
autocontrast_MA_min = min(autocontrast_MA)
autocontrast_MA_min_index = x_axis[autocontrast_MA.index(autocontrast_MA_min)]
autocontrast_MA_mx_coord = 8
autocontrast_MA_my_coord = autocontrast_MA_min+3

# highest value and coordinates
posterize_MA_min = min(posterize_MA)
posterize_MA_min_index = x_axis[posterize_MA.index(posterize_MA_min)]
posterize_MA_mx_coord = 8
posterize_MA_my_coord = posterize_MA_min-3.5


plt.plot(Magnitude, original_MA, label="original", color='red')
plt.scatter(Magnitude, original_MA, c=['red'])

plt.plot(Magnitude, autocontrast_MA, label="autocontrast", color='navy')
plt.scatter(Magnitude, autocontrast_MA, c='navy')
plt.annotate("[A]{}%".format(autocontrast_MA_min), xy=(autocontrast_MA_min_index, autocontrast_MA_min), xytext=(autocontrast_MA_mx_coord,autocontrast_MA_my_coord), **csfont)

plt.plot(Magnitude, brightness_MA, label="brightness", color='#ffed6f')
plt.scatter(Magnitude, brightness_MA, c='#ffed6f')

plt.plot(Magnitude, color_MA, label="color", color='royalblue')
plt.scatter(Magnitude, color_MA, c='royalblue')

plt.plot(Magnitude, contrast_MA, label="contrast", color='#e377c2')
plt.scatter(Magnitude, contrast_MA, c='#e377c2')

plt.plot(Magnitude, equalize_MA, label="equalize", color='#f7b6d2')
plt.scatter(Magnitude, equalize_MA, c='#f7b6d2')

plt.plot(Magnitude, invert_MA, label="invert", color='orangered')
plt.scatter(Magnitude, invert_MA, c='orangered')

plt.plot(Magnitude, posterize_MA, label="posterize", color='sandybrown')
plt.scatter(Magnitude, posterize_MA, c='sandybrown')

plt.plot(Magnitude, rotate_MA, label="rotate", color='tan')
plt.scatter(Magnitude, rotate_MA, c='tan')

plt.plot(Magnitude, sharpness_MA, label="sharpness", color='#6b6ecf')
plt.scatter(Magnitude, sharpness_MA, c='#6b6ecf')

plt.plot(Magnitude, shearX_MA, label="shearX", color='mediumseagreen')
plt.scatter(Magnitude, shearX_MA, c='mediumseagreen')

plt.plot(Magnitude, shearY_MA, label="shearY", color='mediumaquamarine')
plt.scatter(Magnitude, shearY_MA, c='mediumaquamarine')

plt.plot(Magnitude, solarize_MA, label="solarize", color='#31a354')
plt.scatter(Magnitude, solarize_MA, c='#31a354')

plt.plot(Magnitude, translateX_MA, label="translateX", color='orchid')
plt.scatter(Magnitude, translateX_MA, c='orchid')

plt.plot(Magnitude, translateY_MA, label="translateY", color='pink')
plt.scatter(Magnitude, translateY_MA, c='pink')
plt.annotate("[P]{}%".format(posterize_MA_min), xy=(posterize_MA_min_index, posterize_MA_min), xytext=(posterize_MA_mx_coord,posterize_MA_my_coord), **csfont)

plt.legend(bbox_to_anchor=(1.0, 1.0), prop=font)
plt.tight_layout()
# plt.savefig("C:/Users/seacl/Desktop/CIFAR100_linear_MA.png", dpi=200)

plt.show()


Magnitude = [1,3,5,7,9]
plt.xticks(Magnitude, Magnitude)
plt.ylim(0,50)

plt.title("CIFAR-100 Attack accuracy range", **csfont, fontsize=14)
plt.xlabel("Magnitude", **csfont)
plt.ylabel("Accuracy", **csfont)
x_axis = range(0, 15)
# cmap = plt.get_cmap('tab20c') #automatically adjust color by applying color=cmap(i)

# highest value and coordinates
original_AA_max = max(original_AA)
original_AA_max_index = x_axis[original_AA.index(original_AA_max)]
original_AA_mx_coord = 7.9
original_AA_my_coord = original_AA_max+2

# highest value and coordinates
autocontrast_AA_max = max(autocontrast_AA)
autocontrast_AA_max_index = x_axis[autocontrast_AA.index(autocontrast_AA_max)]
autocontrast_AA_mx_coord = 7.9
autocontrast_AA_my_coord = autocontrast_AA_max+2

# highest value and coordinates
posterize_AA_min = min(posterize_AA)
posterize_AA_min_index = x_axis[posterize_AA.index(posterize_AA_min)]
posterize_AA_mx_coord = 8.1
posterize_AA_my_coord = posterize_AA_min-2.2

plt.plot(Magnitude, original_AA, label="original", color='red')
plt.scatter(Magnitude, original_AA, c=['red'])
plt.annotate("[O]{}%".format(original_AA_max), xy=(original_AA_max_index, original_AA_max), xytext=(original_AA_mx_coord,original_AA_my_coord), **csfont)


plt.plot(Magnitude, autocontrast_AA, label="autocontrast", color='navy') 
plt.scatter(Magnitude, autocontrast_AA, c='navy')
plt.annotate("[A]{}%".format(autocontrast_AA_max), xy=(autocontrast_AA_max_index, autocontrast_AA_max), xytext=(autocontrast_AA_mx_coord,autocontrast_AA_my_coord), **csfont)

plt.plot(Magnitude, brightness_AA, label="brightness", color='#ffed6f')
plt.scatter(Magnitude, brightness_AA, c='#ffed6f')

plt.plot(Magnitude, color_AA, label="color", color='royalblue')
plt.scatter(Magnitude, color_AA, c='royalblue')

plt.plot(Magnitude, contrast_AA, label="contrast", color='#e377c2') 
plt.scatter(Magnitude, contrast_AA, c='#e377c2') 

plt.plot(Magnitude, equalize_AA, label="equalize", color='#f7b6d2')
plt.scatter(Magnitude, equalize_AA, c='#f7b6d2')

plt.plot(Magnitude, invert_AA, label="invert", color='orangered')
plt.scatter(Magnitude, invert_AA, c='orangered')

plt.plot(Magnitude, posterize_AA, label="posterize", color='sandybrown')
plt.scatter(Magnitude, posterize_AA, c='sandybrown')

plt.plot(Magnitude, rotate_AA, label="rotate", color='tan')
plt.scatter(Magnitude, rotate_AA, c='tan')

plt.plot(Magnitude, sharpness_AA, label="sharpness", color='#6b6ecf') 
plt.scatter(Magnitude, sharpness_AA, c='#6b6ecf') 

plt.plot(Magnitude, shearX_AA, label="shearX", color='mediumseagreen' )
plt.scatter(Magnitude, shearX_AA, c='mediumseagreen')

plt.plot(Magnitude, shearY_AA, label="shearY", color='mediumaquamarine') 
plt.scatter(Magnitude, shearY_AA, c='mediumaquamarine')

plt.plot(Magnitude, solarize_AA, label="solarize", color='#31a354')
plt.scatter(Magnitude, solarize_AA, c='#31a354')

plt.plot(Magnitude, translateX_AA, label="translateX", color='orchid')
plt.scatter(Magnitude, translateX_AA, c='orchid')

plt.plot(Magnitude, translateY_AA, label="translateY", color='pink')
plt.scatter(Magnitude, translateY_AA, c='pink')
plt.annotate("[P]{}%".format(posterize_AA_min), xy=(posterize_AA_min_index, posterize_AA_min), xytext=(posterize_AA_mx_coord,posterize_AA_my_coord), **csfont)

plt.legend(bbox_to_anchor=(1.0, 1.0), prop=font)
plt.tight_layout()
# plt.savefig("C:/Users/seacl/Desktop/CIFAR100_linear_AA.png", dpi=200)

plt.show()
