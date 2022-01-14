import pandas as pd
import matplotlib.pyplot as plt

# Colors for bar graphs
color_list = ['#31a354', '#31a354', '#e377c2', 'sandybrown', '#ffed6f', '#9edae5', '#f7b6d2', 'sandybrown', '#74c476', '#a1d99b']
augmnetations = ["Solarize (M2)", "Solarize (M2)", "Equalize (M5)", "Equalize (M8)", 
                 "Solarize (M8)", "Solarize (M2)", "Posterize (M7)", "Equalize (M8)",
                 "Brightness (M9)", "Equalize (M8)"]

#model accuracy
labels = ['Aquarium fish', 'Beaver', 'Beetle', 'Boy', 'Can', 'Motorcycle', 'Orchid', 'Rose', 'Sea', 'Tank']
Aug_MA_values = [46.67, 26.67, 56.67, 40, 56.67, 90, 86.67, 46.67, 66.67, 50]
DP_MA_values = [-48,   -11,   -34,-11,   -29,   -64,   -62,   -47,   -62,-53]
CIFAR10_MA_advantage = [-8.00, 15.67, 22.67, 29.00, 17.67, 2.67, 1.33, 0, 4.67, -3]

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

ax.text(-0.5, -59, 'Solarize (M2)', fontsize=12, c='k')
ax.text(-0.18, -54, '-1.67', fontsize=12, c='k')

ax.text(0.46, 34, 'Solarize (M2)', fontsize=12, c='k')
ax.text(0.69, 29, '+15.67', fontsize=12, c='k')

ax.text(1.45, 65, 'Equalize (M5)', fontsize=12, c='k')
ax.text(1.69, 59, '+22.67', fontsize=12, c='k')

ax.text(2.45, 47, 'Equalize (M8)', fontsize=12, c='k')
ax.text(2.73, 42, '+29.0', fontsize=12, c='k')

ax.text(3.45, 64, 'Solarize (M8)', fontsize=12, c='k')
ax.text(3.7, 58, '+27.67', fontsize=12, c='k')

ax.text(4.5, 95, 'Solarize (M2)', fontsize=12, c='k')
ax.text(4.75, 90, '+26.0', fontsize=12, c='k')

ax.text(5.44, 88, 'Posterize (M7)', fontsize=12, c='k')
ax.text(5.7, 82, '+24.67', fontsize=12, c='k')

ax.text(6.45, 53, 'Equalize (M9)', fontsize=12, c='k')
ax.text(6.79, 48, '-0.33', fontsize=12, c='k')

ax.text(7.4, 74, 'Brightness (M9)', fontsize=12, c='k')
ax.text(7.83, 68, '4.67', fontsize=12, c='k')

ax.text(8.42, -64, 'Equalize (M8)', fontsize=12, c='k')
ax.text(8.82, -59, '-3.0', fontsize=12, c='k')

ax.text(1.4, -60, 'Higher MA: Better Utility', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))


#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_MA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_MA_values (colors)', 'DP_MA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.66, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-100 Best Augmentation Advantages Per Label (Model Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
plt.savefig("C:/Users/seacl/Desktop/CIFAR100_MA_advantages.png", dpi=300, bbox_inches = 'tight')



#############################################################################


# attack accuracy
Aug_AA_values = [3.33, 6.67,    10, 3.33, 3.33,   3.33,   20,     20,  6.67,  3.33]
DP_AA_values = [  -20,-6.67, -6.67,  -10,  -10, -23.33, -46.67, -26.67, -6.67, -3.33]
CIFAR10_AA_advantage = [16.67, 0. -3.33, 3.33, 6.67, 20, 40, 6.67, 0, 0]


df_AA = pd.DataFrame({'DP_AA_values': DP_AA_values, 'Aug_AA_values': Aug_AA_values})
fig, ax = plt.subplots(1, figsize = (14, 8))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')  

plt.bar(df_AA.index, df_AA['Aug_AA_values'], color = color_list, width = 0.7)
plt.bar(df_AA.index, df_AA['DP_AA_values'], color = '#1f77b4', width = 0.7)
# plt.plot(df_AA.index, df_AA['difference_AA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-50, 40)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.text(-0.53, 7, 'Solarize (M2)', fontsize=12, c='k')
ax.text(-0.31, 4.3, '+16.67', fontsize=12, c='k')

ax.text(0.43, 10.5, 'Solarize (M2)', fontsize=12, c='k')
ax.text(0.8, 7.5, '±0.0', fontsize=12, c='k')

ax.text(1.45,0 -12, 'Equalize (M5)', fontsize=12, c='k')
ax.text(1.78, -9.5, '-3.33', fontsize=12, c='k')

ax.text(2.45, 7.7, 'Equalize (M8)', fontsize=12, c='k')
ax.text(2.73, 4.5, '+6.67', fontsize=12, c='k')

ax.text(3.4, 4, 'Solarize (M8)', fontsize=12, c='k')
ax.text(3.77, 1, '+6.67', fontsize=12, c='k')

ax.text(4.5, 8, 'Solarize (M2)', fontsize=12, c='k')
ax.text(4.73, 5.5, '+20.0', fontsize=12, c='k')

ax.text(5.44, 24, 'Posterize (M7)', fontsize=12, c='k')
ax.text(5.7, 21, '+26.67', fontsize=12, c='k')

ax.text(6.45, 20.5, 'Equalize (M9)', fontsize=12, c='k')
ax.text(6.75, 18, '+6.67', fontsize=12, c='k')

ax.text(7.4, 10, 'Brightness (M9)', fontsize=12, c='k')
ax.text(7.75, 7.5, '±0.0', fontsize=12, c='k')

ax.text(8.42, 7, 'Equalize (M8)', fontsize=12, c='k')
ax.text(8.8, 4.5, '±0.0', fontsize=12, c='k')

ax.text(1.4, -35, 'Lower AA: Better Privacy', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_AA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_AA_values (colors)', 'DP_AA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.66, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-100 Best Augmentation Advantages Per Label (Attack Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
plt.savefig("C:/Users/seacl/Desktop/CIFAR100_AA_advantages.png", dpi=300, bbox_inches = 'tight')


plt.show()
