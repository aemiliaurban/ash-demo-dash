# # Init
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt
# # Load data
# from sklearn.datasets import load_diabetes
import numpy as np

# Init
import pandas as pd
import scipy

# Clustering
from scipy.cluster.hierarchy import linkage  # You can use SciPy one too
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial import distance

# import seaborn as sns; sns.set()
# Load data
from sklearn.datasets import load_diabetes

#
# # Clustering
# from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
# from scipy.spatial import distance
# from scipy.cluster.hierarchy import linkage # You can use SciPy one too
#
#
# # Dataset
# A_data = load_diabetes().data
# DF_diabetes = pd.DataFrame(A_data, columns = ["attr_%d" % j for j in range(A_data.shape[1])])
#
# # Absolute value of correlation matrix, then subtract from 1 for disimilarity
# DF_dism = 1 - np.abs(DF_diabetes.corr())
#
# # Compute average linkage
# A_dist = distance.squareform(DF_dism)
# Z = linkage(A_dist,method="average")
#
# # Color mapping
# dflt_col = "#808080"   # Unclustered gray
# D_leaf_colors = {"attr_1": dflt_col,
#
#                  "attr_4": "#B061FF", # Cluster 1 indigo
#                  "attr_5": "#B061FF",
#                  "attr_2": "#bfff00",
#                  "attr_8": "#bfff00",
#                  "attr_6": "#bfff00",
#                  "attr_7": "#bfff00",
#
#                  "attr_0": "#ff0000", # Cluster 2 cyan
#                  "attr_3": "#ff0000",
#                  "attr_9": "#ff0000",
#                  }
#
# # notes:
# # * rows in Z correspond to "inverted U" links that connect clusters
# # * rows are ordered by increasing distance
# # * if the colors of the connected clusters match, use that color for link
# link_cols = {}
# for i, i12 in enumerate(Z[:,:2].astype(int)):
#   c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors["attr_%d"%x]
#     for x in i12)
#   link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col
#
# # Dendrogram
# D = dendrogram(Z=Z, labels=DF_dism.index, color_threshold=None,
#   leaf_font_size=12, leaf_rotation=45, link_color_func=lambda x: link_cols[x])
#
# plt.show()


# Dataset
A_data = load_diabetes().data
DF_diabetes = pd.DataFrame(
    A_data, columns=["attr_%d" % j for j in range(A_data.shape[1])]
)

# Absolute value of correlation matrix, then subtract from 1 for disimilarity
DF_dism = 1 - np.abs(DF_diabetes.corr())

# Compute average linkage
A_dist = distance.squareform(DF_dism)
Z = linkage(A_dist, method="average")

# Color mapping
dflt_col = "#808080"  # Unclustered gray
D_leaf_colors = {
    "attr_1": dflt_col,
    "attr_4": "#B061FF",  # Cluster 1 indigo
    "attr_5": "#B061FF",
    "attr_2": "#bfff00",
    "attr_8": "#bfff00",
    "attr_6": "#bfff00",
    "attr_7": "#bfff00",
    "attr_0": "#ff0000",  # Cluster 2 cyan
    "attr_3": "#ff0000",
    "attr_9": "#ff0000",
}


monocrit = np.zeros((Z.shape[0],))
monocrit[[7, 8]] = 1
fc = fcluster(Z, 0, criterion="monocrit", monocrit=monocrit)

for k in range(1, max(fc) + 1):
    print(np.where(fc == k))

# notes:
# * rows in Z correspond to "inverted U" links that connect clusters
# * rows are ordered by increasing distance
# * if the colors of the connected clusters match, use that color for link
link_cols = {}
for i, i12 in enumerate(Z[:, :2].astype(int)):
    c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors["attr_%d" % x] for x in i12)
    link_cols[i + 1 + len(Z)] = c1 if c1 == c2 else dflt_col

# Dendrogram
D = dendrogram(
    Z=Z,
    labels=DF_dism.index,
    color_threshold=None,
    link_color_func=lambda x: link_cols[x],
)


palette = [
    "#800000",
    "#FFD700",
    "#7CFC00",
    "#1E90FF",
    "#D8BFD8",
    "#8B4513",
    "#D8BFD8",
]

# ROUGH EDGES
# IMPLICIT SPLITING
#


def split_dendrogram(linkage_matrix, monocrit):
    # 1 get clusters
    cluster_indices = fcluster(Z, 0, criterion="monocrit", monocrit=monocrit)
    # 2 create color map
    color_map = {cluster: palette[cluster] for cluster in set(cluster_indices)}
    point_color_map = {}
    for nr, p in enumerate(cluster_indices):
        point_color_map[nr] = color_map[p]
    # 3 apply color map
    link_cols = {}
    for i, i12 in enumerate(Z[:, :2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else point_color_map[x] for x in i12)
        link_cols[i + 1 + len(Z)] = c1  #  if c1 == c2 else dflt_col
    return dendrogram(Z=Z, color_threshold=None, link_color_func=lambda x: link_cols[x])


monocrit = np.zeros((Z.shape[0],))
monocrit[[-1, -2, -3, -4, -6]] = 1
print(monocrit)

dendro = split_dendrogram(Z, monocrit)

icoord = scipy.array(dendro["icoord"])
dcoord = scipy.array(dendro["dcoord"])

# seradit podle maxima v dcoord
def sorting_key(item):
    icoord_i, dcoord_i = item
    return max(dcoord_i)


s = sorted(zip(icoord, dcoord), key=sorting_key)
sorted_icoord = [i[0] for i in s]
sorted_dcoord = [i[1] for i in s]


# plt.show()