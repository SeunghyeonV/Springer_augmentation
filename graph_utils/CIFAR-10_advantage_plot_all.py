import pandas as pd
import matplotlib.pyplot as plt

# Colors for bar graphs
color_list = ['#31a354', '#31a354', '#e377c2', 'sandybrown', '#ffed6f', '#9edae5', '#f7b6d2', 'sandybrown', '#74c476', '#a1d99b']

#model accuracy
labels = ['Airplane', 'Automobile', 'Bird', 'cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
Aug_MA_values = [57.4, 80.73, 55, 43.27, 51.8, 69.07, 77.4, 57.73, 49.73, 72.07]
DP_MA_values = [-41.33, -63.8, -32.63, -38.51, -36.17, -46.86, -69.91, -46.25, -47.72, -81.94]

CIFAR10_MA_advantage =[16.07, 16.93, 22.37, 4.76, 15.63, 22.21, 7.49, 11.48, 2.01, -9.87]

df_MA = pd.DataFrame({'DP_MA_values': DP_MA_values, 'Aug_MA_values': Aug_MA_values,
                    'difference_MA' : CIFAR10_MA_advantage})

fig, ax = plt.subplots(1, figsize = (14, 8))
# plt.set_prop_cycle(color=color_list)

plt.bar(df_MA.index, df_MA['Aug_MA_values'], width = 0.6, color=color_list)
plt.bar(df_MA.index, df_MA['DP_MA_values'], color = '#1f77b4', width = 0.6)
plt.plot(df_MA.index, df_MA['difference_MA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-80, 80)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.text(-0.3, 19, '+16.07', fontsize=12, c='k')
ax.text(0.67, 21, '+16.93', fontsize=12, c='k')
ax.text(1.67, 25, '+22.37', fontsize=12, c='k')
ax.text(2.75, 11, '+4.76', fontsize=12, c='k')
ax.text(3.66, 19, '+15.63', fontsize=12, c='k')
ax.text(4.67, 25, '+22.21', fontsize=12, c='k')
ax.text(5.75, 13, '+7.49', fontsize=12, c='k')
ax.text(6.65, 14, '+11.48', fontsize=12, c='k')
ax.text(7.71, 6, '+2.01', fontsize=12, c='k')
ax.text(8.78, -16, '-9.87', fontsize=12, c='w')
ax.text(1.9, -60, 'Higher MA: Better Utility', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))


#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_MA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_MA_values (colors)', 'DP_MA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.75, 1.06, 0, 0]), frameon = True, fontsize=10)

plt.title('CIFAR-10 Best Augmentation Advantages Per Label (Model Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
plt.savefig("C:/Users/seacl/Desktop/MA_advantages.png", dpi=1200)



#############################################################################



# attack accuracy
Aug_AA_values = [22.4, 16.47, 29.67, 23.73, 36.73, 42.53, 57.6, 20.6, 23.6, 40.67]
DP_AA_values = [-34.07, -56.73, -28.47, -34.53, -33.47, -42.8, -68.87, -44.8, -41.73, -78.73]
CIFAR10_AA_advantage = [11.67, 40.26, -1.2, 10.8, -3.26, 0.27, 11.27, 24.2, 18.13, 38.06]
#[11.67, 40.26, -1.2, 10.8, -3.26, 0.27, 11.27, 24.2, 18.13, 38.06]
#[-11.67, -40.26, 1.2, -10.8, 3.26, -0.27, -11.27, -24.2, -18.13, -38.06]

df_AA = pd.DataFrame({'DP_AA_values': DP_AA_values, 'Aug_AA_values': Aug_AA_values,
                    'difference_AA' : CIFAR10_AA_advantage})
fig, ax = plt.subplots(1, figsize = (14, 8))

plt.bar(df_AA.index, df_AA['Aug_AA_values'], color = color_list, width = 0.7)
plt.bar(df_AA.index, df_AA['DP_AA_values'], color = '#1f77b4', width = 0.7)
plt.plot(df_AA.index, df_AA['difference_AA'], color='black', alpha=0.9, marker='.', linewidth=1.5)

# x and y limits
plt.xlim(-0.9, 10)
plt.ylim(-80, 80)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.text(-0.3, 5, '+11.67', fontsize=12, c='k')
ax.text(0.67, 44, '+40.26', fontsize=12, c='k')
ax.text(1.85, -8, '-1.2', fontsize=12, c='w')
ax.text(2.75, 14, '+10.8', fontsize=12, c='k')
ax.text(3.76, -10, '-3.26', fontsize=12, c='w')
ax.text(4.73, 4, '+0.27', fontsize=12, c='k')
ax.text(5.67, 16, '+11.27', fontsize=12, c='k')
ax.text(6.74, 27, '+24.2', fontsize=12, c='k')
ax.text(7.67, 11, '+18.13', fontsize=12, c='k')
ax.text(8.7, 42, '+38.06', fontsize=12, c='k')
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
plt.title('CIFAR-10 Best Augmentation Advantages Per Label (Attack Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
plt.savefig("C:/Users/seacl/Desktop/AA_advantages.png", dpi=1200)


plt.show()
