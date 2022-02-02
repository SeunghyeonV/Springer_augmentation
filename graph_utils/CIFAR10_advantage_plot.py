import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# all font
csfont = {'fontname':'Times New Roman'}
# legend font
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=14)

# Colors for bar graphs
color_list = ['#31a354', '#31a354', '#31a354', '#31a354', '#31a354', '#31a354', '#31a354', '#31a354', '#31a354', '#31a354']
augmentations = ["Solarize (M7)", "Solarize (M7)", "Contrast (M7)", "Posterize (M9)", 
                 "Brightness (M9)", "Equalize (M7)", "Contrast (M6)", "Posterize (M9)",
                 "Solarize (M6)", "Solarize (M7)"]
#model accuracy
labels = ['Airplane', 'Automobile', 'Bird', 'cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
Aug_MA_values = [57.4, 80.73, 55, 43.27, 51.8, 69.07, 77.4, 57.73, 49.73, 63.4]
DP_MA_values = [-41.33, -63.8, -32.63, -38.51, -36.17, -46.86, -69.91, -46.25, -47.72, -81.94]

CIFAR10_MA_advantage =[16.07, 16.93, 22.37, 4.76, 15.63, 22.21, 7.49, 11.48, 2.01, -18.54]

df_MA = pd.DataFrame({'DP_MA_values': DP_MA_values, 'Aug_MA_values': Aug_MA_values})

fig, ax = plt.subplots(1, figsize = (14, 8))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')  

plt.bar(df_MA.index, df_MA['Aug_MA_values'], width = 0.6, color=color_list)
plt.bar(df_MA.index, df_MA['DP_MA_values'], color = '#1f77b4', width = 0.6)
# plt.plot(df_MA.index, df_MA['difference_MA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-100, 100)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.text(-0.45, 65, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(-0.23, 59, '+16.07', fontsize=12, c='k', **csfont)

ax.text(0.5, 88, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(0.75, 82, '+16.93', fontsize=12, c='k', **csfont)

ax.text(1.52, 63, 'Contrast (M7)', fontsize=12, c='k', **csfont)
ax.text(1.75, 57, '+22.37', fontsize=12, c='k', **csfont)

ax.text(2.52, 51, 'Posterize (M9)', fontsize=12, c='k', **csfont)
ax.text(2.8, 45, '+4.76', fontsize=12, c='k', **csfont)

ax.text(3.44, 60, 'Brightness (M9)', fontsize=12, c='k', **csfont)
ax.text(3.75, 54, '+15.63', fontsize=12, c='k', **csfont)

ax.text(4.5, 77, 'Equalize (M7)', fontsize=12, c='k', **csfont)
ax.text(4.75, 71, '+22.21', fontsize=12, c='k', **csfont)

ax.text(5.55, 85, 'Contrast (M6)', fontsize=12, c='k', **csfont)
ax.text(5.8, 79, '+7.49', fontsize=12, c='k', **csfont)

ax.text(6.5, 66, 'Posterize (M9)', fontsize=12, c='k', **csfont)
ax.text(6.75, 60, '+11.48', fontsize=12, c='k', **csfont)

ax.text(7.5, 58, 'Solarize (M6)', fontsize=12, c='k', **csfont)
ax.text(7.8, 52, '+2.01', fontsize=12, c='k', **csfont)

ax.text(8.52, -94, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(8.8, -88, '-18.54', fontsize=12, c='k', **csfont)

ax.text(1.9, -60, 'Higher MA: Better Utility', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'), **csfont)


#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_MA.index , labels = xticks_labels, fontsize=16, **csfont)

# # title and legend
legend_label = ['Augmentation Advantage Scores', 'DP-SGD Advantage Scores']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.78, 1.08, 0, 0]), frameon = True, fontsize=14, prop=font)
plt.title('CIFAR-10 Best Augmentation/Advantages Per Label (Model Accuracy) \n', y=1.01, loc='center', fontdict = {'fontsize' : 24}, **csfont)
# plt.savefig("C:/Users/seacl/Desktop/CIFAR10_MA_advantages.png", dpi=200, bbox_inches = 'tight')



#############################################################################



# attack accuracy
Aug_AA_values = [22.4, 16.47, 29.67, 23.73, 36.73, 42.53, 57.6, 20.6, 23.6, 19.4]
DP_AA_values = [-34.07, -56.73, -28.47, -34.53, -33.47, -42.8, -68.87, -44.8, -41.73, -78.73]
CIFAR10_AA_advantage = [11.67, 40.26, -1.2, 10.8, -3.26, 0.27, 11.27, 24.2, 18.13, 59.33]
#[11.67, 40.26, -1.2, 10.8, -3.26, 0.27, 11.27, 24.2, 18.13, 38.06]
#[-11.67, -40.26, 1.2, -10.8, 3.26, -0.27, -11.27, -24.2, -18.13, -38.06]

df_AA = pd.DataFrame({'DP_AA_values': DP_AA_values, 'Aug_AA_values': Aug_AA_values})
fig, ax = plt.subplots(1, figsize = (14, 8))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')  

plt.bar(df_AA.index, df_AA['Aug_AA_values'], color = color_list, width = 0.7)
plt.bar(df_AA.index, df_AA['DP_AA_values'], color = '#1f77b4', width = 0.7)
# plt.plot(df_AA.index, df_AA['difference_AA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-80, 80)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.text(-0.45, 30, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(-0.25, 25, '+11.67', fontsize=12, c='k', **csfont)

ax.text(0.55, 23, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(0.75, 18, '+40.26', fontsize=12, c='k', **csfont)

ax.text(1.52, -38, 'Contrast (M7)', fontsize=12, c='k', **csfont)
ax.text(1.85, -34, '-1.2', fontsize=12, c='k', **csfont)

ax.text(2.48, 30, 'Posterize (M9)', fontsize=12, c='k', **csfont)
ax.text(2.78, 25, '+10.8', fontsize=12, c='k', **csfont)

ax.text(3.45, -42, 'Brightness (M9)', fontsize=12, c='k', **csfont)
ax.text(3.8, -38, '-3.26', fontsize=12, c='k', **csfont)

ax.text(4.48, 49, 'Equalize (M7)', fontsize=12, c='k', **csfont)
ax.text(4.78, 44, '+0.27', fontsize=12, c='k', **csfont)

ax.text(5.5, 65, 'Contrast (M6)', fontsize=12, c='k', **csfont)
ax.text(5.75, 59, '+11.27', fontsize=12, c='k', **csfont)

ax.text(6.5, 22, 'Posterize (M9)', fontsize=12, c='k', **csfont)
ax.text(6.8, 16, '+24.2', fontsize=12, c='k', **csfont)

ax.text(7.52, 30, 'Solarize (M6)', fontsize=12, c='k', **csfont)
ax.text(7.75, 25, '+18.13', fontsize=12, c='k', **csfont)

ax.text(8.52, 21, 'Solarize (M7)', fontsize=12, c='k', **csfont)
ax.text(8.76, 15, '+59.33', fontsize=12, c='k', **csfont)

ax.text(1.9, -60, 'Lower AA: Better Privacy', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'), **csfont)

#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_AA.index , labels = xticks_labels, fontsize=16, **csfont)

# title and legend
legend_label = ['Augmentation Advantage Scores', 'DP-SGD Advantage Scores']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.78, 1.08, 0, 0]), frameon = True, fontsize=14, prop=font)
plt.title('CIFAR-10 Best Augmentation/Advantages Per Label (Attack Accuracy) \n', y=1.01, loc='center', fontdict = {'fontsize' : 24}, **csfont)
# plt.savefig("C:/Users/seacl/Desktop/CIFAR10_AA_advantages.png", dpi=200, bbox_inches = 'tight')


plt.show()
