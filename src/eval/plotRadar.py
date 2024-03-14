# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data
df = pd.DataFrame({
'group': ['baseline', 'raw','+ emb','+ core','+ emb & core'],
'Recall': [0.1, 0.409, 0.437, 0.448, 0.452],
'F1': [0.1, 0.289, 0.304, 0.211, 0.219],
'Cosine': [0.1, 0.5770, 0.5966, 0.5022, 0.5150],
# 'BLEU': [0, 0, 0, 0, 0],
})
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.15,0.3,0.45], ["0.15","0.3","0.45"], color="grey", size=7)
plt.ylim(0,0.6)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
colors = ['b', 'r', 'g', 'y', 'm']
for i in range(0,len(df)):
    values=df.loc[i].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df['group'][i])
    ax.fill(angles, values, colors[i], alpha=0.1)

 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
# plt.title('Performance of different models')
plt.tight_layout()

# Show the graph
plt.show()
plt.savefig('./figs/radar_plot.png')