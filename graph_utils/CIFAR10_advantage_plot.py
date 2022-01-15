import pandas as pd
import matplotlib.pyplot as plt

# Colors for bar graphs
color_list = ['#31a354', '#31a354', '#e377c2', 'sandybrown', '#ffed6f', '#9edae5', '#f7b6d2', 'sandybrown', '#74c476', '#a1d99b']
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

ax.text(-0.52, 65, 'Solarize (M7)', fontsize=12, c='k')
ax.text(-0.3, 59, '+16.07', fontsize=12, c='k')

ax.text(0.45, 88, 'Solarize (M7)', fontsize=12, c='k')
ax.text(0.67, 82, '+16.93', fontsize=12, c='k')

ax.text(1.45, 63, 'Contrast (M7)', fontsize=12, c='k')
ax.text(1.69, 57, '+22.37', fontsize=12, c='k')

ax.text(2.4, 51, 'Posterize (M9)', fontsize=12, c='k')
ax.text(2.74, 45, '+4.76', fontsize=12, c='k')

ax.text(3.35, 60, 'Brightness (M9)', fontsize=12, c='k')
ax.text(3.66, 54, '+15.63', fontsize=12, c='k')

ax.text(4.4, 77, 'Equalize (M7)', fontsize=12, c='k')
ax.text(4.67, 71, '+22.21', fontsize=12, c='k')

ax.text(5.4, 85, 'Contrast (M6)', fontsize=12, c='k')
ax.text(5.73, 79, '+7.49', fontsize=12, c='k')

ax.text(6.4, 66, 'Posterize (M9)', fontsize=12, c='k')
ax.text(6.65, 60, '+11.48', fontsize=12, c='k')

ax.text(7.45, 58, 'Solarize (M6)', fontsize=12, c='k')
ax.text(7.71, 52, '+2.01', fontsize=12, c='k')

ax.text(8.4, -94, 'Solarize (M7)', fontsize=12, c='k')
ax.text(8.73, -88, '-18.54', fontsize=12, c='k')

ax.text(1.9, -60, 'Higher MA: Better Utility', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))


#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_MA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_MA_values (colors)', 'DP_MA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.66, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-10 Best Augmentation/Advantages Per Label (Model Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/CIFAR10_MA_advantages.png", dpi=300, bbox_inches = 'tight')



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

ax.text(-0.5, 30, 'Solarize (M7)', fontsize=12, c='k')
ax.text(-0.3, 25, '+11.67', fontsize=12, c='k')

ax.text(0.45, 23, 'Solarize (M7)', fontsize=12, c='k')
ax.text(0.67, 18, '+40.26', fontsize=12, c='k')

ax.text(1.45, -39, 'Contrast (M7)', fontsize=12, c='k')
ax.text(1.85, -34, '-1.2', fontsize=12, c='k')

ax.text(2.4, 30, 'Posterize (M9)', fontsize=12, c='k')
ax.text(2.75, 25, '+10.8', fontsize=12, c='k')

ax.text(3.35, -43, 'Brightness (M9)', fontsize=12, c='k')
ax.text(3.76, -38, '-3.26', fontsize=12, c='k')

ax.text(4.4, 50, 'Equalize (M7)', fontsize=12, c='k')
ax.text(4.73, 44, '+0.27', fontsize=12, c='k')

ax.text(5.4, 65, 'Contrast (M6)', fontsize=12, c='k')
ax.text(5.69, 59, '+11.27', fontsize=12, c='k')

ax.text(6.4, 22, 'Posterize (M9)', fontsize=12, c='k')
ax.text(6.74, 16, '+24.2', fontsize=12, c='k')

ax.text(7.45, 30, 'Solarize (M6)', fontsize=12, c='k')
ax.text(7.67, 25, '+18.13', fontsize=12, c='k')

ax.text(8.4, 21, 'Solarize (M7)', fontsize=12, c='k')
ax.text(8.7, 15, '+59.33', fontsize=12, c='k')

ax.text(1.9, -60, 'Lower AA: Better Privacy', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_AA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_AA_values (colors)', 'DP_AA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.66, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-10 Best Augmentation/Advantages Per Label (Attack Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/CIFAR10_AA_advantages.png", dpi=300, bbox_inches = 'tight')


plt.show()

