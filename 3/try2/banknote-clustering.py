import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

# Reading csv file
df = pd.read_csv('banknote-authentication.csv')

# Deleting class column
df = df.drop('class', axis=1)
processed_df = df.to_numpy()[:, :]

# K-means clustering
cluster_num = 2
k_means = KMeans(init='k-means++', n_clusters=cluster_num, n_init=12)
k_means.fit(processed_df)
k_means_labels = k_means.labels_

# Mean-shift clustering
bandwidth = estimate_bandwidth(processed_df, quantile=0.183, n_samples=1000)
m_shift = MeanShift(bandwidth=bandwidth)
m_shift.fit(processed_df)
m_shift_labels = m_shift.labels_
df['ms_cluster'] = m_shift.labels_

# Setting up mpl params
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False
rcParams.update({'figure.autolayout': True})

fig = plt.figure(figsize=(12, 8))

colors = list(map(lambda x: '#002bff' if x == 1 else '#f7ab05', k_means_labels))

ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
#ax1.view_init(azim=0, elev=90)
ax1.scatter(processed_df[:, 0], processed_df[:, 1], processed_df[:, 2], c=colors, alpha=0.8)
ax1.set_zlabel('кратность изображения')
ax1.set_xlabel('дисперсия изображения')
ax1.set_ylabel('асимметрия изображения')
ax1.set_title('K-Means')

colors = list(map(lambda x: '#002bff' if x == 1 else '#f7ab05', m_shift_labels))

ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
#ax2.view_init(azim=0, elev=90)
ax2.scatter(processed_df[:, 0], processed_df[:, 1], processed_df[:, 2], c=colors, alpha=0.8)
ax2.set_zlabel('кратность изображения')
ax2.set_xlabel('дисперсия изображения')
ax2.set_ylabel('асимметрия изображения')
ax2.set_title('Mean-shift')

# images showing and saving
plt.savefig('banknote-authentication_clustering_done.svg')
plt.show()