import math
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from plotly.graph_objs import graph_objs
from scipy.cluster.hierarchy import fcluster

from common.data_parser import RDataParser
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified

matplotlib.pyplot.switch_backend("agg")
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["r", "k", "c"])

r = RDataParser()
r.convert_merge_matrix()
r.add_joining_height()

default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=default_cycler)


def plot_input_data_reduced(plot_input_data: str, plot_master: PlotMaster):
    if plot_input_data == "All dimensions":
        return plot_master.plot_all_dimensions()
    elif plot_input_data == "PCA":
        return plot_master.plot_pca()
    elif "PCA_3D" in plot_input_data:
        return plot_master.plot_pca(dimensions=3)
    elif plot_input_data == "tSNE":
        return plot_master.plot_tsne()
    elif plot_input_data == "tSNE_3D":
        return plot_master.plot_tsne(dimensions=3)
    elif plot_input_data == "UMAP":
        return plot_master.plot_umap()
    elif plot_input_data == "UMAP_3D":
        return plot_master.plot_umap(dimensions=3)
    else:
        return plot_master.plot_pca()


def extract_highest_x(data):
    highest_x = float('-inf')  # Initialize with a very small value

    for d in data:
        x_values = d['x']
        max_x = max(x_values)
        if max_x > highest_x:
            highest_x = max_x

    return highest_x


def assign_clusters(points):
    clusters = []
    current_cluster = []
    prev_color = None

    for point_id, color in points.items():
        if prev_color is None or color != prev_color:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []

        current_cluster.append((point_id, color))
        prev_color = color

    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def convert_to_dict(clusters):
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        point_ids = [point for point in cluster]
        cluster_dict[str(i)] = point_ids
    return cluster_dict

def calculate_cluster_percentages(data):
    total_length = 0

    for lst in data.values():
        total_length += len(lst)

    cluster_counts = {}

    cluster_percentages = {}
    # Count the occurrences of each cluster
    for i in range(len(list(data.values()))):
        cluster_percentages[f"{i}"] = (len(list(data.values())[i]) / total_length) * 100

    return cluster_percentages


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div(
    [
        html.Header("Dendrogram", style={"fontSize": 40}),
        html.H6("Choose color threshold for the dendrogram.", style={"fontSize": 25, "margin-top": "25px"}),
        dcc.Slider(
            min=0,
            max=r.max_tree_height,
            value=10,
            marks=None,
            id="color-threshold-slider",
        ),
        dcc.Dropdown(
            ["Colorblind palette on", "Colorblind palette off"],
            multi=False,
            id="colorblind-palette-dropdown",
        ),
        dcc.Graph(id="dendrogram-graph", figure=go.Figure()),
        html.Div(id='no-of-clusters-output'),
        html.Div(id="clusters"),
        dcc.RadioItems(id="ClusterRadio", options=[], value=''),
        html.Header("Heatmap", style={"fontSize": 40, "margin-top": "25px"}),
        dcc.Dropdown(list(r.dataset.columns), multi=True, id="dropdown-heatmap-plot", value="All"),
        dcc.Graph(id="heatmap-graph", figure=go.Figure()),
        html.Header("Two features plot", style={"fontSize": 40, "margin-top": "25px"}),
        dcc.Dropdown(
            list(r.dataset.columns), multi=True, id="dropdown-selected-features-plot"
        ),
        dcc.Graph(id="two-features", figure=go.Figure()),
        html.Header(
            "Dimension reduction plot", style={"fontSize": 40, "margin-top": "25px"}
        ),
        dcc.Dropdown(
            [
                "All dimensions",
                "PCA",
                "PCA_3D",
                "tSNE",
                "tSNE_3D",
                "UMAP",
                "UMAP_3D",
            ],
            id="plot_dropdown",
            value=["PCA"],
        ),
        dcc.Graph(id="reduced-graph", figure=go.Figure()),
        dcc.Store(id="dendrogram_memory"),
    ]
)


@app.callback(
    Output("dendrogram_memory", "data"),
    Input("color-threshold-slider", "value"),
    Input("colorblind-palette-dropdown", "value"),
)
def create_dendrogram(value, colorblind_palette_input):
    colorblind_palette = (
        True if colorblind_palette_input == "Colorblind palette on" else False
    )
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        custom_dendrogram = create_dendrogram(
            r.merge_matrix,
            color_threshold=value,
            labels=r.labels,
            colorblind_palette=colorblind_palette,
        )
        assigned_clusters = convert_to_dict(assign_clusters(custom_dendrogram.leaves_color_map_translated))
        to_return = {
            "leaves_color_map_translated": custom_dendrogram.leaves_color_map_translated,
            "clusters": custom_dendrogram.clusters,
            "labels": custom_dendrogram.labels,
            "data": custom_dendrogram.data,
            "layout": custom_dendrogram.layout,
            "color_threshold": value,
            "icoord": custom_dendrogram.xvals,
            "dcoord": custom_dendrogram.yvals,
            "assigned_clusters": assigned_clusters
        }
        return to_return


@app.callback(
    Output("dendrogram-graph", "figure"),
    Input("dendrogram_memory", "data"),
)
def plot_dendrogram(data):
    fig = graph_objs.Figure(data=data["data"], layout=data["layout"])
    fig.add_shape(
        type="line",
        x0=0,
        y0=data["color_threshold"],
        x1=extract_highest_x(data["data"]),
        y1=data["color_threshold"],
        line=dict(color="red", width=2, dash="dash"),
    )
    return fig


@app.callback(
    Output("no-of-clusters-output", "children"),
    Input("dendrogram_memory", "data")
)
def get_number_of_clusters(data):
    return f"Number of clusters: {data['clusters']}"


@app.callback(
    Output("clusters", "children"),
    Input("dendrogram_memory", "data")
)
def get_cluster_percentages(data):
    clusters = data["assigned_clusters"]
    cluster_percentages = calculate_cluster_percentages(clusters)

    return f"Individual cluster percentages: {cluster_percentages}"

# Callback to update options
@app.callback(
    Output('ClusterRadio', 'options'),
    Input("dendrogram_memory", "data")
)
def update_options(data):
    options = []
    for i in range(len(data["assigned_clusters"].keys())):
        options.append({"label": html.Div([list(data["assigned_clusters"].keys())[i]],
                                          style={'color': list(data["assigned_clusters"].items())[i][1][1][1],
                                                 'font-size': 20}),
                        "value": f"{i}"})
    return options


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("dropdown-heatmap-plot", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_heatmap(value, data):
    plot_master = PlotMaster(
        r.dataset, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    if value == "All":
        fig_heatmap = go.Figure(
            data=go.Heatmap(plot_master.df_to_plotly(r.dataset, r.dataset.columns))
        )
    elif type(value) != list or len(value) < 2:
        fig_heatmap = go.Figure(
            data=go.Heatmap(plot_master.df_to_plotly(r.dataset, r.dataset.columns[0:2]))
        )
    else:
        fig_heatmap = go.Figure(
            data=go.Heatmap(plot_master.df_to_plotly(r.dataset, value))
        )
    return fig_heatmap


@app.callback(
    Output("two-features", "figure"),
    Input("dropdown-selected-features-plot", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_two_selected_features(value, data):
    plot_master = PlotMaster(
        r.dataset, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    if type(value) != list or len(value) != 2:
        feature_plot = go.Figure(
            plot_master.plot_selected_features(r.dataset.columns[0:2])
        )
    else:
        feature_plot = go.Figure(plot_master.plot_selected_features(value))
    return feature_plot


@app.callback(
    Output("reduced-graph", "figure"),
    Input("plot_dropdown", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_data_reduced(value, data):
    plot_master = PlotMaster(
        r.dataset, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    return plot_input_data_reduced(value, plot_master)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
