import pandas as pd
import matplotlib.pyplot as plt

# Colors for bar graphs
color_list = ['#31a354', '#31a354', '#e377c2', 'sandybrown', '#ffed6f', '#9edae5', '#f7b6d2', 'sandybrown', '#74c476', '#a1d99b']

#model accuracy
labels = ['Aquarium fish', 'Beaver', 'Beetle', 'Boy', 'Can', 'Motorcycle', 'Orchid', 'Rose', 'Sea', 'Tank']
Aug_MA_values = [40, 26.67, 56.67, 40, 46.67, 66.67, 63.33, 46.67, 66.67, 50]
DP_MA_values = [-48,   -11,   -34,-11,   -29,   -64,   -62,-46.67,   -62,-53]
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
plt.ylim(-90, 90)

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.text(-0.18, -54, '-8.0', fontsize=12, c='k')
ax.text(0.69, 29, '+15.67', fontsize=12, c='k')
ax.text(1.69, 59, '+22.67', fontsize=12, c='k')
ax.text(2.73, 42, '+29.0', fontsize=12, c='k')
ax.text(3.68, 48, '+17.67', fontsize=12, c='k')
ax.text(4.75, 69, '+2.67', fontsize=12, c='k')
ax.text(5.75, 65, '+1.33', fontsize=12, c='k')
ax.text(6.77, 48, '+0.0', fontsize=12, c='k')
ax.text(7.83, +69, '4.67', fontsize=12, c='k')
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
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.75, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-100 Best Augmentation Advantages Per Label (Model Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/MA_advantages.png", dpi=300, bbox_inches = 'tight')



#############################################################################



# attack accuracy
Aug_AA_values = [3.33, 6.67, 10, 6.67, 3.33, 3.33, 6.67, 20, 6.67, 3.33]
DP_AA_values = [-20, -6.67, -6.67, -10, -10, -23.33, -46.67, -26.67, -6.67, -3.33]
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
ax.text(-0.31, 4.5, '+16.67', fontsize=12, c='k')
ax.text(0.8, 8, '+0.0', fontsize=12, c='k')
ax.text(1.78, -9.5, '-3.33', fontsize=12, c='k')
ax.text(2.73, 8, '+3.33', fontsize=12, c='k')
ax.text(3.77, 5, '+6.67', fontsize=12, c='k')
ax.text(4.76, 5, '+20.0', fontsize=12, c='k')
ax.text(5.75, 8, '+40.0', fontsize=12, c='k')
ax.text(6.75, 21, '+6.67', fontsize=12, c='k')
ax.text(7.75, 7.5, '+0.0', fontsize=12, c='k')
ax.text(8.8, 4.5, '+0.0', fontsize=12, c='k')
ax.text(1.4, -35, 'Lower AA: Better Privacy', fontsize=12, c='k', size='xx-large', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

#grid
ax.set_axisbelow(False)
# ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

# x ticks
xticks_labels = labels
plt.xticks(df_AA.index , labels = xticks_labels, fontsize=13)

# title and legend
legend_label = ['Advantages', 'Aug_AA_values (colors)', 'DP_AA_values']
plt.legend(legend_label, ncol = 3, bbox_to_anchor=([0.75, 1.06, 0, 0]), frameon = True, fontsize=10)
plt.title('CIFAR-100 Best Augmentation Advantages Per Label (Attack Accuracy) \n', loc='center', fontdict = {'fontsize' : 20})
# plt.savefig("C:/Users/seacl/Desktop/AA_advantages.png", dpi=300, bbox_inches = 'tight')


plt.show()

# total_advantage_score = []
