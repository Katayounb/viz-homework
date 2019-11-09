import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data/breast-cancer/wdbc.data',
                           sep=',',
                           header=0)

df.columns=['id', 'diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness', 'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension']

df.drop('id', axis=1, inplace=True)

#print(df)

plt.tight_layout()
os.makedirs('plots/breast-cancer', exist_ok=True)

# All the plots in this section use the breast cancer dataset as base

# Example of creating a Histogram plot
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.hist(df['mean symmetry'], bins=30, color='g', label='mean symmetry')
axes.set_title('Mean Simmetry')
axes.set_xlabel('Buckets')
axes.set_ylabel('Mean Simmetry')
axes.legend()
plt.savefig('plots/breast-cancer/cancer_mean_simmetry_hist.png', dpi=300)

# Example of creating a Pie plot with percentage (autopct)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.pie(df['diagnosis'].value_counts(), autopct='%1.1f%%', labels=df['diagnosis'].value_counts().index.tolist())
axes.set_title('Diagnosis')
axes.legend()
plt.savefig('plots/breast-cancer/cancer_diagnosis_pie.png', dpi=300)

# Scatter plot
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(df['mean area'], df['mean fractal dimension'], s=5, c='blue', alpha=0.2, marker='D')
axes.set_title('mean area VS mean fractal dimension')
axes.set_xlabel('mean area')
axes.set_ylabel('mean fractal dimension')
axes.grid(True)
plt.savefig('plots/breast-cancer/cancer_scatter.png', dpi=300)


#scatter example for all "mean" vs "error"
for column1 in (['mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']):
    for column2 in (['compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error']):
            print(f' {column1} VS {column2} plot')
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            axes.scatter(df[column1], df[column2], label=f'{column1} VS {column2}', color='green', marker='x')
            axes.set_title(f'{column1} VS {column2}')
            axes.set_xlabel(column1)
            axes.set_ylabel(column2)
            axes.legend()
            plt.savefig(f'plots/breast-cancer/breast_{column1}_vs_{column2}_scatter.png', dpi=300)
            plt.close(fig)

# Example of creating a Correlation Heatmap plot, based on this, the following have close Correlation
# 'radius error' and 'area error'
# 'mean radius'  and  'mean perimeter'
# 'mean radius' and 'mean area'
# 'worst radius' and 'worst perimeter'
# 'worst radius' and 'mean area'
# 'mean radius' and 'worst perimeter'
# fig, axes = plt.subplots(1, 1, figsize=(20, 20))
# df['encoded_diagnosis']=df['diagnosis'].map({'B': 0, 'M': 1})
# correlation = df.corr().round(2)
# im = axes.imshow(correlation)
# cbar = axes.figure.colorbar(im, ax=axes)
# cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
# numrows = len(correlation.iloc[0])
# numcolumns = len(correlation.columns)
# axes.set_xticks(np.arange(numrows))
# axes.set_yticks(np.arange(numcolumns))
# axes.set_xticklabels(correlation.columns)
# axes.set_yticklabels(correlation.columns)
# plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
# for i in range(numrows):
#     for j in range(numcolumns):
#         text = axes.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='w')
# axes.set_title('Heatmap of Correlation of Dimensions')
# fig.tight_layout()
# plt.savefig('plots/breast-cancer/cancer_correlation_heatmap.png')

# Example with seaborn joinplot - kind is optional
sns.jointplot('worst texture', 'worst symmetry', data=df, kind='reg')
plt.savefig(f'plots/breast-cancer/breast_cancer_joinplot.png', dpi=300)

# Example with seaborn lineplot
sns.lineplot('worst texture', 'worst area', hue="diagnosis", data=df)
plt.savefig(f'plots/breast-cancer/breast_cancer_lineplot.png', dpi=300)


fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sns.heatmap(df.corr(), cbar=True, yticklabels=df.columns, xticklabels=df.columns)
plt.savefig(f'plots/breast-cancer/corr_heatmap.png', dpi=300)


# Example of creating a 3D plot
malign = df[df['diagnosis'] == 'M']
benign = df[df['diagnosis'] == 'B']
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
line1 = axes.scatter(malign['worst perimeter'], malign['worst area'], malign['worst concavity'])
line2 = axes.scatter(benign['worst perimeter'], benign['worst area'], benign['worst concavity'])
axes.legend((line1, line2), ('Malign', 'Benign'))
axes.set_xlabel('worst perimeter')
axes.set_ylabel('worst area')
axes.set_zlabel('worst concavity')
plt.savefig('plots/breast-cancer/3d-scatter.png')


# Try to make 3D plot with tridsurf
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_trisurf(df['worst perimeter'], df['worst area'], df['worst concavity'], color='blue', antialiased=True)
axes.set_xlabel('worst perimeter')
axes.set_ylabel('worst area')
axes.set_zlabel('worst concavity')
plt.savefig('plots/breast-cancer/3d-trisurf.png')


plt.close()