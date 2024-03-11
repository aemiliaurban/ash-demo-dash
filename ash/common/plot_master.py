import os

import numpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

DATA_FOLDER = os.path.join(os.getcwd(), "common", "user_data")

REDUCED_DIMENSIONS_FOLDER = "reduced_dimensions"

numpy.set_printoptions(threshold=9999999999999)  # TODO: INSPECT consider np.inf


class PlotMaster:
    def __init__(
        self, input_data, labels: list[str], order: list[int | float], color_map: dict
    ):
        self.input_data = input_data
        self.labels = labels
        self.order = order
        self.color_map = color_map

    def plot_dendrogram(self, dendrogram):
        return go.Figure(data=dendrogram.data, layout=dendrogram.layout)

    def order_labels(self):
        ordered_labels = []
        for index in self.order:
            ordered_labels.append(self.labels[int(index)])
        return ordered_labels

    def df_to_plotly(
        self,
        df: pd.DataFrame,
        desired_columns: list,
    ):
        df = df[desired_columns].T
        return {"z": df.values, "y": df.index.tolist()}

    def plot_all_dimensions(self):
        features = self.input_data.columns
        fig = px.scatter_matrix(
            self.input_data,
            dimensions=features,
            color=list(self.color_map.keys()),
            color_discrete_map=self.color_map,
            hover_name=self.labels,
        )
        fig.update_traces(diagonal_visible=True)
        return fig

    def plot_pca(self, color_map, dimensions: int = 2):
        filename = "pca.txt" if dimensions == 2 else "pca_3D.txt"
        pca = self.read_reduction(filename, REDUCED_DIMENSIONS_FOLDER)
        if not pca:
            pca = PCA(dimensions).fit_transform(self.input_data)
            # self.save_reduction(pca, filename, REDUCED_DIMENSIONS_FOLDER)

        if dimensions == 2:
            fig = px.scatter(
                pca,
                x=0,
                y=1,
                title="First and Second Principal Components",
                color=color_map,
                color_discrete_map={c: c for c in color_map},
            )
            fig.update_layout(
                {
                    "xaxis_title": "1st Component",
                    "yaxis_title": "2nd Component",
                    "showlegend": False,
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                    ]
                ),
                name="",
            )
        else:
            fig = px.scatter_3d(
                pca,
                x=0,
                y=1,
                z=2,
                color=color_map,
                color_discrete_map={c: c for c in color_map},
                title="First Three Principal Components",
            )
            fig.update_layout(
                {
                    "showlegend": False,
                    "scene": {
                        "xaxis_title": "1st Component",
                        "yaxis_title": "2nd Component",
                        "zaxis_title": "3rd Component",
                    },
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                        "3rd Component: %{z}",
                    ]
                ),
                name="",
            )
        return fig

    def plot_tsne(self, color_map, dimensions: int = 2):
        filename = "tsne.txt" if dimensions == 2 else "tsne_3D.txt"
        tsne = self.read_reduction(filename, REDUCED_DIMENSIONS_FOLDER)
        if not tsne:
            tsne = TSNE(
                n_components=dimensions, random_state=0, perplexity=5
            ).fit_transform(self.input_data)
            # self.save_reduction(tsne, filename, REDUCED_DIMENSIONS_FOLDER)
        if dimensions == 2:
            fig = px.scatter(
                tsne,
                x=0,
                y=1,
                title="t-distributed Stochastic Neighbor Embedding (t-SNE) - First Two Components",
                color=color_map,
                color_discrete_map={c: c for c in color_map},
            )
            fig.update_layout(
                {
                    "xaxis_title": "1st Component",
                    "yaxis_title": "2nd Component",
                    "showlegend": False,
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                    ]
                ),
                name="",
            )
        else:
            fig = px.scatter_3d(
                tsne,
                x=0,
                y=1,
                z=2,
                color=color_map,
                color_discrete_map={c: c for c in color_map},
                title="t-distributed Stochastic Neighbor Embedding (t-SNE) - First Three Components",
            )
            fig.update_layout(
                {
                    "showlegend": False,
                    "scene": {
                        "xaxis_title": "1st Component",
                        "yaxis_title": "2nd Component",
                        "zaxis_title": "3rd Component",
                    },
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                        "3rd Component: %{z}",
                    ]
                ),
                name="",
            )
        return fig

    def plot_umap(self, color_map, dimensions: int = 2):
        filename = "umap.txt" if dimensions == 2 else "umap_3D.txt"
        umap = self.read_reduction(filename, REDUCED_DIMENSIONS_FOLDER)
        if not umap:
            umap = UMAP(
                n_components=dimensions, init="random", random_state=0
            ).fit_transform(self.input_data)
            # self.save_reduction(umap, filename, REDUCED_DIMENSIONS_FOLDER)
        if dimensions == 2:
            fig = px.scatter(
                umap,
                x=0,
                y=1,
                title="Uniform Manifold Approximation and Projection (UMAP) - First Two Components",
                color=color_map,
                color_discrete_map={c: c for c in color_map},
            )
            fig.update_layout(
                {
                    "xaxis_title": "1st Component",
                    "yaxis_title": "2nd Component",
                    "showlegend": False,
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                    ]
                ),
                name="",
            )
        else:
            fig = px.scatter_3d(
                umap,
                x=0,
                y=1,
                z=2,
                color=color_map,
                color_discrete_map={c: c for c in color_map},
                title="Uniform Manifold Approximation and Projection (UMAP) - First Three Components",
            )
            fig.update_layout(
                {
                    "showlegend": False,
                    "scene": {
                        "xaxis_title": "1st Component",
                        "yaxis_title": "2nd Component",
                        "zaxis_title": "3rd Component",
                    },
                }
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "1st Component: %{x}",
                        "2nd Component: %{y}",
                        "3rd Component: %{z}",
                    ]
                ),
                name="",
            )
        return fig

    def plot_selected_features(self, desired_columns, color_mask):
        to_plot = self.input_data[desired_columns]

        fig = go.Figure(
            go.Scatter(
                x=to_plot.loc[:, to_plot.columns[0]],
                y=to_plot.loc[:, to_plot.columns[1]],
                mode="markers",
                marker_color=color_mask,
            )
        )

        return fig

    @staticmethod
    def save_reduction(
        data, filename: str, subfolder: str, path_to_folder: str = DATA_FOLDER
    ) -> None:
        with open(os.path.join(path_to_folder, subfolder, filename), "w") as file:
            file.write(str(data))

    @staticmethod
    def read_reduction(
        filename: str, subfolder: str, path_to_folder: str = DATA_FOLDER
    ) -> list[list[float]] | None:
        data = []
        try:
            with open(os.path.join(path_to_folder, subfolder, filename), "r") as file:
                for line in file.readlines():
                    row = list(
                        map(
                            float,
                            line.strip().replace("[", "").replace("]", "").split(),
                        )
                    )
                    data.append(row)
        except FileNotFoundError:
            return None
        return data

    @staticmethod
    def update_marker_color(fig, trace_index, new_color):
        trace = fig.data
        trace.marker.color = new_color
        fig.data[trace_index] = trace

        return fig