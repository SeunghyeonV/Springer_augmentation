import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# all font
csfont = {'fontname':'Times New Roman'}
# legend font
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

labels = ['Aquarium fish', 'Beaver', 'Beetle', 'Boy', 'Can', 'Motorcycle', 'Orchid', 'Rose', 'Sea', 'Tank']

augmnetations = ["Solarize (M3)", "Solarize (M2)", "Equalize (M5)", "Equalize (M8)", 
                 "Posterize (M9)", "Solarize (M8)", "Posterize (M9)", "Equalize (M9)",
                 "Brightness (M9)", "Posterize (M7)"]


advantage_scores = np.array([[8.67, -1.0, -20.66, 2.33, 27.66, 42.66, 11.34, -13.66, -35.33, -9.67], 
                            [15.34, 15.67, -17.33, 12.34, 24.33, 46.0, 18.01, -13.66, -21.99, -13.0], 
                            [5.34, 5.67, 19.34, 32.33, 21.0, 45.99, 48.0, 3.01, -12.0, -16.34], 
                            [5.34, 2.33, 2.67, 35.67, 14.34, 42.66, 44.67, 3.01, -25.33, -16.34], 
                            [-11.33, -4.33, -24.0, 2.33, 24.34, 29.33, 41.33, -3.66, -45.33, -36.34], 
                            [-11.33, -4.33, -20.66, 2.33, 34.34, 22.67, -2.0, -13.66, -35.33, -36.34], 
                            [-11.33, -4.33, -24.0, 2.33, 24.34, 29.33, 41.33, -3.66, -45.33, -36.34], 
                            [5.34, -1.0, 16.01, 32.33, 17.67, 35.99, 38.0, 6.34, -8.67, -19.67], 
                            [-4.67, -1.0, -17.33, 12.34, -2.34, 2.67, 11.34, -17.0, 4.67, -36.33], 
                            [12.0, -1.0, -20.67, 15.67, 17.66, 46.0, 51.34, 3.0, -52.0, -3.0]])

advantage_scores = advantage_scores.T
# print(advantage_scores_t)

best_advantage_scores = []
for i in range(10):
    best_advantage_scores.append(max(advantage_scores[i]))
# print(best_advantage_scores)


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(advantage_scores)

# Setting the labels
ax.set_xticks(np.arange(len(augmnetations)))
ax.set_yticks(np.arange(len(labels)))
# labeling respective list entries
ax.set_xticklabels(augmnetations, fontsize=20, **csfont)
ax.set_yticklabels(labels, fontsize=20, **csfont)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

# Creating text annotations by using for loop
for i in range(len(augmnetations)):
    for j in range(len(labels)):
        text = ax.text(j, i, advantage_scores[i, j],
                        ha="center", va="center", color="k", fontsize=16)
        # if advantage_scores[i, j] < 0:
        #     text = ax.text(j, i, advantage_scores[i, j],
        #         ha="center", va="center", color="w", fontsize=16)
        if advantage_scores[i, j] in best_advantage_scores:
            text = ax.text(j, i, advantage_scores[i, j],
                ha="center", va="center", color="red", fontsize=16)
            best_advantage_scores.remove(advantage_scores[i, j])
            
            
# Calculate (height_of_image / width_of_image)
# im_ratio = im.shape[0]/im.shape[1]

ax.set_title("CIFAR-100 Advantage Scores", fontsize=28, y=1.02, **csfont)
fig.tight_layout()
cbar = plt.colorbar(im, orientation='vertical', fraction=0.045)
# cbar.set_label('Advantage Scores', size=20)

# plt.savefig("C:/Users/seacl/Desktop/CIFAR100_matrix.png", dpi=200, bbox_inches = 'tight')
plt.show()
