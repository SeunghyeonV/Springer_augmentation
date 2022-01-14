import pandas as pd
import matplotlib.pyplot as plt

# # Colors for bar graphs
color_list = ['#31a354', '#31a354', '#e377c2', 'sandybrown', '#ffed6f', '#9edae5', '#f7b6d2', 'sandybrown', '#74c476', '#a1d99b']

#model accuracy
labels = ['Airplane', 'Automobile', 'Bird', 'cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
Aug_MA_values = [69.73, 85, 42, 43.27, 31.73, 42.6, 69.4, 57.73, 43.4, 64.73]
DP_MA_values = [-41.33, -63.8, -32.63, -38.51, -36.17, -46.86, -69.91, -46.25, -47.72, -81.94]

CIFAR10_MA_advantage = [28.4, 21.2, 9.37, 4.76, -4.44, -4.26, -0.51, 11.48, -4.32, -17.21]

df_MA = pd.DataFrame({'DP_MA_values': DP_MA_values, 'Aug_MA_values': Aug_MA_values,
                    'difference_MA' : CIFAR10_MA_advantage})

fig, ax = plt.subplots(1, figsize = (14, 8))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')  

plt.bar(df_MA.index, df_MA['Aug_MA_values'], width = 0.6, color='sandybrown')
plt.bar(df_MA.index, df_MA['DP_MA_values'], color = '#1f77b4', width = 0.6)
plt.plot(df_MA.index, df_MA['difference_MA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-90, 90)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.text(-0.28, 32, '+28.4', fontsize=12, c='k')
ax.text(0.74, 25, '+21.2', fontsize=12, c='k')
ax.text(1.74, 14, '+9.37', fontsize=12, c='k')
ax.text(2.73, 9, '+4.76', fontsize=12, c='k')
ax.text(3.77, -12, '-4.44', fontsize=12, c='w')
ax.text(4.78, -12, '-4.26', fontsize=12, c='w')
ax.text(5.78, -9, '-0.51', fontsize=12, c='w')
ax.text(6.69, 14, '+11.48', fontsize=12, c='k')
ax.text(7.78, -13, '-4.32', fontsize=12, c='w')
ax.text(8.72, -25, '-17.21', fontsize=12, c='w')
ax.text(1.9, -60, 'Higher MA: Better Utility', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))


#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_MA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_MA_values', 'DP_MA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.7, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-10 Posterize Advantages Per Label (Model Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/Posterize_MA_advantages.png", dpi=600, bbox_inches = 'tight')



#############################################################################



# attack accuracy
Aug_AA_values = [40.6, 27, 22.47, 23.73, 35.4, 21.07, 78.66, 20.6, 26.27, 44.2]
DP_AA_values = [-34.07, -56.73, -28.47, -34.53, -33.47, -42.8, -68.87, -44.8, -41.73, -78.73]
CIFAR10_AA_advantage = [-6.53, 29.73, 6.0, 10.8, -1.93, 21.73, -9.79, 24.2, 15.46, 34.53]


df_AA = pd.DataFrame({'DP_AA_values': DP_AA_values, 'Aug_AA_values': Aug_AA_values,
                    'difference_AA' : CIFAR10_AA_advantage})
fig, ax = plt.subplots(1, figsize = (14, 8))
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1')  

plt.bar(df_AA.index, df_AA['Aug_AA_values'], color = 'sandybrown', width = 0.7)
plt.bar(df_AA.index, df_AA['DP_AA_values'], color = '#1f77b4', width = 0.7)
plt.plot(df_AA.index, df_AA['difference_AA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-90, 90)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.text(-0.23, -14, '-6.53', fontsize=12, c='w')
ax.text(0.67, 33, '+29.73', fontsize=12, c='k')
ax.text(1.83, 11, '+6.0', fontsize=12, c='k')
ax.text(2.75, 14, '+10.8', fontsize=12, c='k')
ax.text(3.77, -10, '-1.93', fontsize=12, c='w')
ax.text(4.7, 26, '+21.73', fontsize=12, c='k')
ax.text(5.78, -18, '-9.79', fontsize=12, c='w')
ax.text(6.74, 27, '+24.2', fontsize=12, c='k')
ax.text(7.67, 22, '+15.46', fontsize=12, c='k')
ax.text(8.7, 38, '+34.53', fontsize=12, c='k')
ax.text(1.9, -60, 'Lower AA: Better Privacy', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_AA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_AA_values (colors)', 'DP_AA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.75, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-10 Posterize Advantages Per Label (Attack Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/Posterize_AA_advantages.png", dpi=600, bbox_inches = 'tight')


plt.show()

total_advantage_score = [21.87, 50.93, 15.37, 15.56, -6.37, 17.47, -10.3, 35.68, 11.14, 17.32]

