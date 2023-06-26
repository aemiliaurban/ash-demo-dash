from unittest.mock import patch

import dash_bootstrap_components as dbc
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ash.common.util import convert_to_dict, assign_clusters, extract_lowest_and_highest_x, \
    create_point_position_dictionary, get_elements_from_list, plot_input_data_reduced, calculate_cluster_percentages
from common.data_parser import RDataParser
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified
from dash import Dash, Input, Output, dcc, html
from plotly.graph_objs import graph_objs

matplotlib.pyplot.switch_backend("agg")
r = RDataParser()
r.convert_merge_matrix()
r.add_joining_height()


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div(
    [
        html.Header("Dendrogram", style={"fontSize": 40}),
        html.H6(
            "Choose color threshold for the dendrogram.",
            style={"fontSize": 25, "margin-top": "25px"},
        ),
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
        html.Div(id="no-of-clusters-output"),
        html.Div(id="clusters"),
        dcc.RadioItems(id="ClusterRadio", options=[], value=""),
        html.Header("Heatmap", style={"fontSize": 40, "margin-top": "25px"}),
        dcc.Dropdown(
            list(r.dataset.columns), multi=True, id="dropdown-heatmap-plot", value="All"
        ),
        dcc.Graph(id="heatmap-graph", figure=go.Figure()),
        html.Header("Two features plot", style={"fontSize": 40, "margin-top": "25px"}),
        html.Div(id="error-message"),
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
        assigned_clusters = convert_to_dict(
            assign_clusters(custom_dendrogram.leaves_color_map_translated)
        )
        to_return = {
            "leaves_color_map_translated": custom_dendrogram.leaves_color_map_translated,
            "clusters": custom_dendrogram.clusters,
            "labels": custom_dendrogram.labels,
            "data": custom_dendrogram.data,
            "layout": custom_dendrogram.layout,
            "color_threshold": value,
            "icoord": custom_dendrogram.xvals,
            "dcoord": custom_dendrogram.yvals,
            "assigned_clusters": assigned_clusters,
        }
        return to_return


@app.callback(
    Output("dendrogram-graph", "figure"),
    [Input("dendrogram_memory", "data"), Input("ClusterRadio", "value")],
)
def plot_dendrogram(data, highlight_area):
    fig = graph_objs.Figure(data=data["data"], layout=data["layout"])
    _, highest_x_data = extract_lowest_and_highest_x(data["data"])
    fig.add_shape(
        type="line",
        x0=0,
        y0=data["color_threshold"],
        x1=highest_x_data,
        y1=data["color_threshold"],
        line=dict(color="red", width=2, dash="dash"),
    )

    highlight_area_points = []
    if highlight_area:
        highlight_area_points_and_colors = data["assigned_clusters"][highlight_area]
        highlight_area_points = [
            point_color[0] for point_color in highlight_area_points_and_colors
        ]
        point_position_dictionary = create_point_position_dictionary(data["labels"])
        used_positions = {
            key: value
            for key, value in point_position_dictionary.items()
            if key in highlight_area_points
        }
        trimmed_data = get_elements_from_list(
            data["data"],
            list(used_positions.values())[: len(list(used_positions.values())) - 1],
        )
        lowest_x, highest_x = extract_lowest_and_highest_x(trimmed_data)
        #
        # fig.add_shape(
        #     type="rect",
        #     xref="x",
        #     yref="paper",
        #     x0=lowest_x,  # Use the start and end points of highlight area
        #     y0=0,
        #     x1=highest_x,
        #     y1=1,
        #     fillcolor="rgba(255,0,0,0.2)",  # Set the background color for the highlight area
        #     layer="below",
        #     line_width=0,
        # )
    for i, point in enumerate(fig.data):
        point.hovertext = data["labels"][i]

        if highlight_area and data["labels"][i] in highlight_area_points:
            point.fillcolor = "red"
            point.marker.color = "red"

    return fig


@app.callback(
    Output("no-of-clusters-output", "children"), Input("dendrogram_memory", "data")
)
def get_number_of_clusters(data):
    return f"Number of clusters: {data['clusters']}"


@app.callback(Output("clusters", "children"), Input("dendrogram_memory", "data"))
def get_cluster_percentages(data):
    clusters = data["assigned_clusters"]
    cluster_percentages = calculate_cluster_percentages(clusters)

    return f"Individual cluster percentages: {cluster_percentages}"


# Callback to update options
@app.callback(Output("ClusterRadio", "options"), Input("dendrogram_memory", "data"))
def update_options(data):
    options = []
    for i in range(len(data["assigned_clusters"].keys())):
        options.append(
            {
                "label": html.Div(
                    [list(data["assigned_clusters"].keys())[i]],
                    style={
                        "color": list(data["assigned_clusters"].items())[i][1][1][1],
                        "font-size": 20,
                    },
                ),
                "value": f"{i}",
            }
        )
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
    Output("error-message", "children"),
    Input("dropdown-selected-features-plot", "value"),
    Input("dendrogram_memory", "data"),
)
def plot_two_selected_features(value, data):
    if type(value) != list or len(value) != 2:
        # Return an empty figure if number of selected values is not two
        feature_plot = go.Figure()
        error_message = "Please select exactly two values."
    else:
        plot_master = PlotMaster(
            r.dataset, data["labels"], r.order, data["leaves_color_map_translated"]
        )
        feature_plot = plot_master.plot_selected_features(value)
        error_message = None

    return feature_plot, error_message


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
