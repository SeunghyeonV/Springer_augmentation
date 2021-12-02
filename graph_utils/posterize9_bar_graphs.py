import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

labels = ['Airplane', 'Automobile', 'Bird', 'cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
DP_values = [41.33, 63.8, 32.63, 38.51, 36.17, 46.86, 69.91, 46.25, 47.72, 81.94]
Aug_values = [69.73, 85, 42, 43.27, 31.73, 42.6, 69.4, 57.73, 43.4, 64.73]


x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, DP_values, width, label='A group', color='#a9a9a9')


cmap = plt.get_cmap('tab20')
# ax.set_prop_cycle(color=[cmap(k) for k in x])
ax.set_prop_cycle(color='sandybrown')
for i in x:
    subbars_bar2 = ax.bar(i+width/2, Aug_values[i], width=width)
    # ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    if i == 0:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 1:
        ax.bar_label(subbars_bar2, padding=1, fontsize=8)
    elif i == 2:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 3:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 4:
        ax.bar_label(subbars_bar2, padding=-1, fontsize=8)
    elif i == 5:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 6:
        ax.bar_label(subbars_bar2, padding=-10, fontsize=8)
    elif i == 8:
        ax.bar_label(subbars_bar2, padding=-1, fontsize=8)
    else:
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('DP-SGD (σ=0.5) & Posterize(M9) Model accuracy chart')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.text(5.7, 105, 'Gray bars: DP-SGD (σ=0.5)', fontsize=10,  color='black')
ax.text(5.75, 95, 'Color bars: Posterize (M9)', fontsize=10,  color='black')

ax.bar_label(bar1, padding=0, fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.ylim(0, 115)
fig.tight_layout()
plt.show()




labels = ['Airplane', 'Automobile', 'Bird', 'cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
DP_values = [34.07, 56.73, 28.47, 34.53, 33.47, 42.8, 68.87, 44.8, 41.73, 78.73]
Aug_values = [40.6, 27, 22.47, 23.73, 35.4, 21.07, 78.66, 20.6, 26.27, 44.2]


x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, DP_values, width, label='A group', color='#a9a9a9')


cmap = plt.get_cmap('tab20')
# ax.set_prop_cycle(color=[cmap(k) for k in x])
ax.set_prop_cycle(color='sandybrown')

for i in x:
    subbars_bar2 = ax.bar(i+width/2, Aug_values[i], width=width)
    # ax.bar_label(subbars_bar2, padding=3, fontsize=8)
    if i == 0:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 1:
        ax.bar_label(subbars_bar2, padding=1, fontsize=8)
    elif i == 2:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 3:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 4:
        ax.bar_label(subbars_bar2, padding=5, fontsize=8)
    elif i == 5:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 6:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    elif i == 8:
        ax.bar_label(subbars_bar2, padding=0, fontsize=8)
    else:
        ax.bar_label(subbars_bar2, padding=3, fontsize=8)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('DP-SGD (σ=0.5) & Posterize(M9) Attack accuracy chart')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.text(5.7, 105, 'Gray bars: DP-SGD (σ=0.5)', fontsize=10,  color='black')
ax.text(5.75, 95, 'Color bars: Posterize (M9)', fontsize=10,  color='black')

ax.bar_label(bar1, padding=0, fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.ylim(0, 115)
fig.tight_layout()
plt.show()
