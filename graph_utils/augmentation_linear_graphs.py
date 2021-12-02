import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## Draw linear graphs for each augmentation that shows accuracy change per magnitude


autocontrast_MA = [82.31, 81.93, 82.36, 82.37, 81.92]
brightness_MA = [84.07, 81.02, 73.86, 55.13, 44.15]
color_MA = [83.82, 83.27, 82.61, 81.73, 80.11]
contrast_MA = [82.92, 81.84, 76.25, 61.93, 48.27]
equalize_MA = [69.79, 69.14, 69.54, 70.17,69.99]
invert_MA = [39.57, 39.53, 39.81, 39.23, 39.33]
posterize_MA = [83.35, 83.83, 84.01, 76.98, 54.96]
rotate_MA = [81.13, 76.42, 66.75, 57.25, 48.07]
sharpness_MA = [84.35, 83.29, 82.93, 81.61, 80.48]
shearX_MA = [82.95, 80.72, 76.01, 70.48, 63.56]
shearY_MA = [83.47, 81.83, 76.95, 70.14, 63.63]
solarize_MA = [70.93, 60.60, 58.95, 48.1, 40.23]
translateX_MA = [80.53, 66.08, 51.01, 38.17, 30.87]
translateY_MA = [80.71, 67.21, 49.22, 35.71, 24.45]

autocontrast_AA = [73.02, 71.65, 71.72, 74.42, 71.77] 
brightness_AA = [73, 68.81, 56.17, 43.22, 40.63]
color_AA = [74.64, 71.45, 71.40, 69.50, 70.16]
contrast_AA = [75.48, 69.16, 59.20, 45.77, 36.62]
equalize_AA = [58.42, 58.70, 57.07, 55.63, 57.72]
invert_AA = [30.49, 31.13, 28.87, 28.86, 30.72]
posterize_AA = [73.31, 73.34, 74.18, 63.56, 34]
rotate_AA = [72.43, 66.89, 58.95, 45.27, 39.42]
sharpness_AA = [72.45, 73.15, 72.92, 68.77, 67.43]
shearX_AA = [72.91, 80.72, 76.01, 70.48, 63.57]
shearY_AA = [72.43, 71.12, 69.13, 59.26, 52.15]
solarize_AA = [53.77, 42.21, 44.95, 25.81, 31.02]
translateX_AA = [70.24, 57.64, 42.37, 31.38, 23.93]
translateY_AA = [70.42, 58.65, 42.78, 29.57, 19.84]


Magnitude = [1,3,5,7,9]
plt.xticks(Magnitude, Magnitude)
plt.ylim(10,90)

plt.title("Model accuracy range based on magnitude")
plt.xlabel("Magnitude")
plt.ylabel("Accuracy")
x_axis = range(0, 15)
# cmap = plt.get_cmap('tab20c') #automatically adjust color by applying color=cmap(i)

autocontrast_MA_min = min(autocontrast_MA)
autocontrast_MA_min_index = x_axis[autocontrast_MA.index(autocontrast_MA_min)]
autocontrast_MA_mx_coord = 8.1
autocontrast_MA_my_coord = autocontrast_MA_min+2


translateY_MA_min = min(translateY_MA)
translateY_MA_min_index = x_axis[translateY_MA.index(translateY_MA_min)]
translateY_MA_mx_coord = 8.1
translateY_MA_my_coord = translateY_MA_min-5


plt.plot(Magnitude, autocontrast_MA, label="autocontrast", color='navy')
plt.scatter(Magnitude, autocontrast_MA)
plt.annotate("{}%".format(autocontrast_MA_min), xy=(autocontrast_MA_min_index, autocontrast_MA_min), xytext=(autocontrast_MA_mx_coord,autocontrast_MA_my_coord))

plt.plot(Magnitude, brightness_MA, label="brightness", color='#ffed6f')
plt.scatter(Magnitude, brightness_MA)

plt.plot(Magnitude, color_MA, label="color", color='royalblue')
plt.scatter(Magnitude, color_MA)

plt.plot(Magnitude, contrast_MA, label="contrast", color='#e377c2')
plt.scatter(Magnitude, contrast_MA)

plt.plot(Magnitude, equalize_MA, label="equalize", color='#f7b6d2')
plt.scatter(Magnitude, equalize_MA)

plt.plot(Magnitude, invert_MA, label="invert", color='orangered')
plt.scatter(Magnitude, invert_MA)

plt.plot(Magnitude, posterize_MA, label="posterize", color='sandybrown')
plt.scatter(Magnitude, posterize_MA)

plt.plot(Magnitude, rotate_MA, label="rotate", color='tan')
plt.scatter(Magnitude, rotate_MA)

plt.plot(Magnitude, sharpness_MA, label="sharpness", color='#6b6ecf')
plt.scatter(Magnitude, sharpness_MA)

plt.plot(Magnitude, shearX_MA, label="shearX", color='mediumseagreen')
plt.scatter(Magnitude, shearX_MA)

plt.plot(Magnitude, shearY_MA, label="shearY", color='mediumaquamarine')
plt.scatter(Magnitude, shearY_MA)

plt.plot(Magnitude, solarize_MA, label="solarize", color='#31a354')
plt.scatter(Magnitude, solarize_MA)

plt.plot(Magnitude, translateX_MA, label="translateX", color='orchid')
plt.scatter(Magnitude, translateX_MA)

plt.plot(Magnitude, translateY_MA, label="translateY", color='pink')
plt.scatter(Magnitude, translateY_MA)
plt.annotate("{}%".format(translateY_MA_min), xy=(translateY_MA_min_index, translateY_MA_min), xytext=(translateY_MA_mx_coord,translateY_MA_my_coord))

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("C:/Users/seacl/Desktop/MA_linear.png", dpi=1200)

plt.show()


Magnitude = [1,3,5,7,9]
plt.xticks(Magnitude, Magnitude)
plt.ylim(10,90)

plt.title("Attack accuracy range based on magnitude")
plt.xlabel("Magnitude")
plt.ylabel("Accuracy")
x_axis = range(0, 15)
# cmap = plt.get_cmap('tab20c') #automatically adjust color by applying color=cmap(i)

autocontrast_AA_min = min(autocontrast_AA)
autocontrast_AA_min_index = x_axis[autocontrast_AA.index(autocontrast_AA_min)]
autocontrast_AA_mx_coord = 8.1
autocontrast_AA_my_coord = autocontrast_AA_min+3

translateY_AA_min = min(translateY_AA)
translateY_AA_min_index = x_axis[translateY_AA.index(translateY_AA_min)]
translateY_AA_mx_coord = 8.1
translateY_AA_my_coord = translateY_AA_min-5


plt.plot(Magnitude, autocontrast_AA, label="autocontrast", color='navy') 
plt.scatter(Magnitude, autocontrast_AA)
plt.annotate("{}%".format(autocontrast_AA_min), xy=(autocontrast_AA_min_index, autocontrast_AA_min), xytext=(autocontrast_AA_mx_coord,autocontrast_AA_my_coord))

plt.plot(Magnitude, brightness_AA, label="brightness", color='#ffed6f')
plt.scatter(Magnitude, brightness_AA)

plt.plot(Magnitude, color_AA, label="color", color='royalblue')
plt.scatter(Magnitude, color_AA)

plt.plot(Magnitude, contrast_AA, label="contrast", color='#e377c2') 
plt.scatter(Magnitude, contrast_AA)

plt.plot(Magnitude, equalize_AA, label="equalize", color='#f7b6d2')
plt.scatter(Magnitude, equalize_AA)

plt.plot(Magnitude, invert_AA, label="invert", color='orangered')
plt.scatter(Magnitude, invert_AA)

plt.plot(Magnitude, posterize_AA, label="posterize", color='sandybrown')
plt.scatter(Magnitude, posterize_AA)

plt.plot(Magnitude, rotate_AA, label="rotate", color='tan')
plt.scatter(Magnitude, rotate_AA)

plt.plot(Magnitude, sharpness_AA, label="sharpness", color='#6b6ecf') 
plt.scatter(Magnitude, sharpness_AA)

plt.plot(Magnitude, shearX_AA, label="shearX", color='mediumseagreen' )
plt.scatter(Magnitude, shearX_AA)

plt.plot(Magnitude, shearY_AA, label="shearY", color='mediumaquamarine') 
plt.scatter(Magnitude, shearY_AA)

plt.plot(Magnitude, solarize_AA, label="solarize", color='#31a354')
plt.scatter(Magnitude, solarize_AA)

plt.plot(Magnitude, translateX_AA, label="translateX", color='orchid')
plt.scatter(Magnitude, translateX_AA)

plt.plot(Magnitude, translateY_AA, label="translateY", color='pink')
plt.scatter(Magnitude, translateY_AA)
plt.annotate("{}%".format(translateY_AA_min), xy=(translateY_AA_min_index, translateY_AA_min), xytext=(translateY_AA_mx_coord,translateY_AA_my_coord))

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("C:/Users/seacl/Desktop/AA_linear.png", dpi=1200)

plt.show()
