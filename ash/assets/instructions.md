# Instructions

Ash is interactive tool for clustering unstructured data. It targets flow cytometry data, but the tool itself is data agnostic.
In flow cytometry, the leading method to cluster data is called manual gating and it relies on researchers to manually draw boundaries around clusters of cells. This is a time consuming and subjective process.
Ash aims to facilitate the process by providing tools and visualization to do so.

Recently, there has been progress in GPU accelerated clustering algorithms and Ash serves as a front end to these.
User is expected to provide a raw data (linkage matrix - result of clustering) and assign clusters to different subgroups in Ash.
Ash itself is not a tool for performing clustering, but rather a tool for distinguishing cellular populations and visualizing them.  

# Expected Input Data Format
Out of the box, ash comes with sample data, but it is easy to replace them via the web interface.
However, the data must be in the following format: ...

# How to use Ash
In Ash a dendrogram is deployed and user is expected to input points where the dendrogram should be split.
To dedice about the splitpoints, series of supportive plots are provided.

## Splitting the Dendrogram
In the main screen user can input numerical label of split point, click the split button and the dendrogram will be split at that point, provided that implicit splitting rule is not violated. For details see the section below.
The rest of the plots are updated accordingly.

#### Notes on Nodes
Nodes of dendrogram are labeled according to their height, such that the highest node is labeled with 1, second highest with 2 and so on.
From now on, we will refer to nodes by their labels. See the figure below for reference.

![Ash] (assets/enumeration.svg)
#### Implicit Splitting
Let's assume user is interested in splitting the data at Node 3. That would mean that leaves B and C would form their clusters, but what about leaves A and D?
Are they really part of the same cluster? According to this dendrogram, distance between A and D would be the greatest among all pairs of the leaves.
Apart from that in more complex examples, such splitting could lead to ambiguous cluster assignments for some data points.

![Ash] (assets/implicit.svg)

Ash does not allow such splitting and will not do it implicitly for user.
If user want to split the dendrogram at Node 4, she must also split at Nodes 1 and 2. Otherwise, Ash won't split the dendrogram for her.

In other word if you take any split point in the tree and traverse to the root, all visited nodes on the path must be split points as well.

## Heatmap
Note, that heatmap only shows the data points in the cluster user selected in the table.



