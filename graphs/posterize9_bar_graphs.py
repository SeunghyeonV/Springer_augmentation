import numpy as np
import matplotlib
import matplotlib.pyplot as plt

total_psnr = [4.84, 9.46, 5.63, 2.62, 18.16, 2.35, 10.97, 6.05, 4.98, 2.65]
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

x = np.arange(len(labels)) 
width = 0.5

fig, ax = plt.subplots()
ax.set_ylabel('Mean PSNR')
ax.set_title('Mean PSNR of reconstructed dataset')

cmap = plt.get_cmap('tab10')
for i in x:
    if i == 0 or i == 3 or i == 5 or i == 8:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='dodgerblue', label='Posterize [M9]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    elif i == 1:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='darkorange', label='Brightness [M5]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    elif i == 2:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='moccasin', label='Equalize [M7]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    elif i == 4:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='darkgreen', label='TranslateY [M9]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    elif i == 6:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='limegreen', label='TranslateY [M3]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    elif i == 7:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='tan', label='Equalize [M9]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    else:
        subbars_bar2 = ax.bar(i, total_psnr[i], width=width, color='lightpink', label='Solarize [M7]')
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xticks(rotation=45, ha='right')

plt.ylim(0, 20)
fig.tight_layout()

plt.legend(bbox_to_anchor=(1.35, 1.0))
plt.show()
