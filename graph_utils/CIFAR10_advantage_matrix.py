import numpy as np
import matplotlib
import matplotlib.pyplot as plt

labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog",
          "Horse", "Ship", "Truck"]
augmnetations = ["Solarize (M7)", "Solarize (M7)", "Contrast (M7)", "Posterize (M9)", 
                 "Brightness (M9)", "Equalize (M7)", "Contrast (M6)", "Posterize (M9)",
                 "Solarize (M6)", "Solarize (M3)"]


advantage_scores =  np.array([[27.74, 57.19, 14.57, 8.69, -4.37, 17.14, -14.84, 24.15, 10.88, 40.79], 
                    [27.74, 57.19, 14.57, 8.69, -4.37, 17.14, -14.84, 24.15, 10.88, 40.79], 
                    [17.34, 20.66, 21.17, 8.49, -9.43, 11.54, 18.22, 22.08, 7.81, 2.85], 
                    [21.87, 50.93, 15.37, 15.56, -6.37, 17.47, -10.3, 35.68, 11.14, 17.32], 
                    [6.34, -46.2, 0.64, -18.11, 12.37, 6.2, 8.56, 16.42, -0.86, 8.92], 
                    [20.67, 26.53, 10.11, 2.42, -8.76, 22.48, -9.71, 29.68, 5.41, 5.66], 
                    [16.21, 29.06, 16.5, 5.02, -1.24, 11.4, 18.76, 20.09, 1.87, 12.59], 
                    [21.87, 50.93, 15.37, 15.56, -6.37, 17.47, -10.3, 35.68, 11.14, 17.32], 
                    [24.21, 57.0, 11.51, 2.62, -9.1, 9.94, 5.42, 23.95, 20.14, 21.46], 
                    [13.34, 42.73, 6.11, 1.69, -4.03, 16.08, 6.49, 23.22, 9.14, 28.19]])


advantage_scores_t = advantage_scores.T
# print(advantage_scores_t)

best_advantage_scores = []
for i in range(10):
    best_advantage_scores.append(max(advantage_scores_t[i]))
# print(best_advantage_scores)


fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(advantage_scores)

# Setting the labels
ax.set_xticks(np.arange(len(augmnetations)))
ax.set_yticks(np.arange(len(labels)))
# labeling respective list entries
ax.set_xticklabels(augmnetations, fontsize=20)
ax.set_yticklabels(labels, fontsize=20)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

# Creating text annotations by using for loop
for i in range(len(augmnetations)):
    for j in range(len(labels)):
        text = ax.text(j, i, advantage_scores_t[i, j],
                        ha="center", va="center", color="k", fontsize=16)
        if advantage_scores_t[i, j] < 0:
            text = ax.text(j, i, advantage_scores_t[i, j],
                ha="center", va="center", color="w", fontsize=16)
        if advantage_scores_t[i, j] in best_advantage_scores:
            text = ax.text(j, i, advantage_scores_t[i, j],
                ha="center", va="center", color="red", fontsize=16)
            
            
# Calculate (height_of_image / width_of_image)
# im_ratio = im.shape[0]/im.shape[1]

ax.set_title("CIFAR-10 Advantage Scores", fontsize=28, y=1.02)
fig.tight_layout()
cbar = plt.colorbar(im, orientation='vertical', fraction=0.045)
cbar.set_label('Advantage Score', size=20)

# plt.savefig("C:/Users/seacl/Desktop/CIFAR10_Advantage_score_maxtrix.png", dpi=300, bbox_inches = 'tight')
plt.show()
