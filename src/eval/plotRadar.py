import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# (baseline), raw, +emb, +core, +emb+core
recall = []
F1 = []
cosine_similarity = []
bleu = []

df = pd.DataFrame({
    'recall': np.array(recall),
    'F1': np.array(F1),
    'cos': np.array(cosine_similarity),
    'bleu': np.array(bleu)
})

# Convert the DataFrame into long format
df_long = pd.melt(df, var_name='metrics', value_name='values')

# Create a subplot with polar projection
fig, ax = plt.subplots(subplot_kw=dict(polar=True))

# Draw a line plot on the polar axis
sns.lineplot(x='metrics', y='values', data=df_long, ax=ax)

# Set the yticks to be invisible
plt.yticks([])

# Set the xticks to be the metrics names
plt.xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False), ['recall', 'F1', 'cosine similarity', 'bleu'])

plt.show()
# save the plot
plt.savefig('radar_plot.png')
