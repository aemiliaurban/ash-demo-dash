import copy
from unittest.mock import patch

import dash
import dash_bootstrap_components as dbc
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ash.common.color_mappings import C_TO_NAME_COLORMAP
from common.data_parser import RDataParser
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified
from dash import Dash, Input, Output, dcc, html, State
from plotly.graph_objs import graph_objs

from ash.common.util import (
    assign_clusters,
    calculate_cluster_percentages,
    convert_to_dict,
    extract_lowest_and_highest_x,
    plot_input_data_reduced,
    write_to_text_file,
    read_text_file,
    modify_dendrogram_color,
    replace_color_values,
    get_click_coordinates,
    parse_value_string,
)

matplotlib.pyplot.switch_backend("agg")
r = RDataParser()

r.parse()


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
        html.Header("Dendrogram with customizable clustering", style={"fontSize": 40}),
        dcc.Input(
            id="range-input",
            type="text",
            placeholder="Enter range [min-x, max-x, min-y, max-y]",
            style={"width": "400px"},
        ),
        dcc.Input(id="color-input", type="text", value="#ff0000"),
        html.Button("Accept", id="accept-button", n_clicks=0),
        html.Div(id="output-div"),
        html.Div(id="click-output", children=""),
        html.Div(id="min-output"),
        html.Div(id="max-output"),
        html.Button("Assign Min", id="assign-min-button", n_clicks=0),
        html.Button("Assign Max", id="assign-max-button", n_clicks=0),
        dcc.Input(id="color-input-manual", type="text", value="#ff0000"),
        html.Button("Accept", id="accept-button-manual", n_clicks=0),
        html.Div(id="output-div-manual"),
        dcc.Graph(id="custom-dendrogram-graph", figure=go.Figure()),
        html.Header("Heatmap", style={"fontSize": 40, "margin-top": "25px"}),
        html.Div(
            id="heatmap-message",
            children=["Displaying a subset of the data due to constraints"],
            style={"color": "black", "font-size": 20, "margin-bottom": "10px"},
        ),
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
            "Dimensionality reduction plot", style={"fontSize": 40, "margin-top": "25px"}
        ),
        html.Div(id="error-message-dim-red"),
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
            value=None,
        ),
        dcc.Graph(id="reduced-graph", figure=go.Figure()),
        html.Div(id="error-message-dim-custom"),
        dcc.Graph(id="reduced-graph-custom", figure=go.Figure()),
        dcc.Store(id="dendrogram-memory"),
        dcc.Store(id="custom-dendrogram-color-map"),
        dcc.Store(id="click-values-output", data=[]),
    ]
)


"""
Dendrogram initialization
"""


@app.callback(
    Output("dendrogram-memory", "data"),
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
        write_to_text_file("custom_dendrogram.txt", custom_dendrogram.dendro)
        to_return = {
            "dendro": custom_dendrogram.dendro,
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


"""
Dendrogram related stuff
"""


@app.callback(
    Output("dendrogram-graph", "figure"),
    [Input("dendrogram-memory", "data"), Input("ClusterRadio", "value")],
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

    for i, point in enumerate(fig.data):
        point.hovertext = data["labels"][i]

        if data["labels"][i] in highlight_area_points:
            point.fillcolor = "red"
            point.marker.color = "red"

    return fig


@app.callback(
    Output("no-of-clusters-output", "children"), Input("dendrogram-memory", "data")
)
def get_number_of_clusters(data):
    return f"Number of clusters: {data['clusters']}"


@app.callback(Output("clusters", "children"), Input("dendrogram-memory", "data"))
def get_cluster_percentages(data):
    clusters = data["assigned_clusters"]
    cluster_percentages = calculate_cluster_percentages(clusters)

    return f"Individual cluster percentages: {cluster_percentages}"


@app.callback(Output("ClusterRadio", "options"), Input("dendrogram-memory", "data"))
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


"""
User controlled dendrogram
"""


@app.callback(
    Output("output-div", "children"),
    Input("accept-button", "n_clicks"),
    State("range-input", "value"),
    State("color-input", "value"),
)
def update_output(n_clicks, range_value, color):
    if n_clicks > 0 and range_value:
        try:
            min_x, max_x, min_y, max_y = map(float, range_value[1:-1].split(","))
            dendro = read_text_file("custom_dendrogram.txt")
            modify_dendrogram_color(dendro, min_x, max_x, min_y, max_y, color)
            write_to_text_file("custom_dendrogram.txt", dendro)
            return html.H4(
                f"Range: [min-x: {min_x}, max-x: {max_x}, min-y: {min_y}, max-y: {max_y}]"
            )
        except ValueError:
            return html.H4(
                "Please enter a valid range in the format [min-x, max-x, min-y, max-y]"
            )
    else:
        return html.H4(
            "Please enter a range and click Accept or click on the graph and use assign buttons"
        )


@app.callback(
    Output("custom-dendrogram-graph", "figure"),
    Output("custom-dendrogram-color-map", "data"),
    Input("accept-button", "n_clicks"),
    Input("dendrogram-memory", "data"),
    Input("accept-button-manual", "n_clicks"),
)
def plot_dendrogram_custom(clicks, orig_dendrogram, clicks_manual):
    dendrogram = read_text_file("custom_dendrogram.txt")
    dendrogram = replace_color_values(dendrogram, C_TO_NAME_COLORMAP)

    traces = []
    print(len(dendrogram["icoord"]))
    print(len(dendrogram["dcoord"]))
    print(len(dendrogram["color_list"]))
    print(len(dendrogram["ivl"]))
    print(len(dendrogram["leaves"]))
    print(len(dendrogram["leaves_color_list"]))

    for xs, ys, color, label in zip(
        dendrogram["icoord"],
        dendrogram["dcoord"],
        dendrogram["color_list"],
        dendrogram["ivl"],
    ):
        trace = go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color),
            showlegend=False,
            hovertext=label,
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(clickmode="event")
    fig.data[0].on_click(get_click_coordinates)
    custom_dendrogram_color_map = dict()
    print(len(fig.data))
    for point in fig.data:
        custom_dendrogram_color_map[point["hovertext"]] = point["line"]["color"]

    return (fig, custom_dendrogram_color_map)


@app.callback(
    Output("click-output", "children"),
    Output("click-values-output", "data"),
    Input("custom-dendrogram-graph", "clickData"),
)
def handle_click(click_data):
    if click_data:
        x = click_data["points"][0]["x"]
        y = click_data["points"][0]["y"]
        return f"Clicked on point at x = {x}, y = {y}", [x, y]
    else:
        return "", []


@app.callback(
    Output("min-output", "children"),
    Input("assign-min-button", "n_clicks"),
    Input("click-values-output", "data"),
    State("min-output", "children"),
    prevent_initial_call=True,
)
def handle_min_click(n_clicks, click_output, current_output):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_output

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "assign-min-button" and n_clicks > 0 and click_output:
        n_clicks = 0
        return f"min_x, min_y = {click_output}"

    return current_output


@app.callback(
    Output("max-output", "children"),
    Input("assign-max-button", "n_clicks"),
    State("click-values-output", "data"),
    State("max-output", "children"),
    prevent_initial_call=True,
)
def handle_max_click(n_clicks, click_output, current_output):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_output

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "assign-max-button" and n_clicks > 0 and click_output:
        n_clicks = 0
        return f"max_x, max_y = {click_output}"

    return current_output


@app.callback(
    Output("output-div-manual", "children"),
    Input("accept-button-manual", "n_clicks"),
    State("min-output", "children"),
    State("max-output", "children"),
    State("color-input-manual", "value"),
)
def update_output(n_clicks, min_output, max_output, color):
    if n_clicks > 0 and min_output and max_output:
        try:
            min_x, min_y = map(float, parse_value_string(min_output))
            max_x, max_y = map(float, parse_value_string(max_output))

            if max_x < min_x or max_y < min_y:
                raise ValueError(
                    "Max values must be greater than or equal to min values"
                )

            dendro = read_text_file("custom_dendrogram.txt")
            modify_dendrogram_color(dendro, min_x, max_x, min_y, max_y, color)
            write_to_text_file("custom_dendrogram.txt", dendro)

            return html.H4(
                f"Range: [min-x: {min_x}, max-x: {max_x}, min-y: {min_y}, max-y: {max_y}]"
            )
        except ValueError as e:
            return html.H4(
                "Please enter valid numeric values for min-x, min-y, max-x, and max-y"
            )
    else:
        return html.H4("")


"""
Heatmap stuff 
"""


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("dropdown-heatmap-plot", "value"),
    Input("dendrogram-memory", "data"),
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


"""
Other plots
"""


@app.callback(
    Output("two-features", "figure"),
    Output("error-message", "children"),
    Input("dropdown-selected-features-plot", "value"),
    Input("dendrogram-memory", "data"),
    Input("ClusterRadio", "value"),
)
def plot_two_selected_features(value, data, highlight_area):
    if type(value) != list or len(value) != 2:
        feature_plot = go.Figure()
        error_message = "Please select exactly two values."
    else:
        color_map = copy.deepcopy(data["leaves_color_map_translated"])
        if highlight_area:
            highlight_area_points_and_colors = data["assigned_clusters"][highlight_area]
            highlight_area_points = [
                point_color[0] for point_color in highlight_area_points_and_colors
            ]
            for point in highlight_area_points:
                color_map[point] = "red"
        plot_master = PlotMaster(r.dataset, data["labels"], r.order, color_map)
        feature_plot = plot_master.plot_selected_features(value)
        error_message = None
    return feature_plot, error_message


@app.callback(
    Output("reduced-graph", "figure"),
    Output("error-message-dim-red", "children"),
    Input("plot_dropdown", "value"),
    Input("dendrogram-memory", "data"),
    Input("ClusterRadio", "value"),
)
def plot_data_reduced(value, data, highlight_area):
    if value is None:
        reduced_plot = go.Figure()
        error_message = (
            "Please select dimensionality reduction.\n "
            "Ash will attempt to access pre-calculated dimensionality reduction data at the designated location (ash/common/user_data/reduced_dimensions)."
            "If the data is not found, "
            "Ash will perform the necessary calculations on the matrix data provided at the designated location (ash/common/user_data/data.csv), "
            "which may result in longer processing time."
        )
    else:
        color_map = copy.deepcopy(data["leaves_color_map_translated"])
        if highlight_area:
            highlight_area_points_and_colors = data["assigned_clusters"][highlight_area]
            highlight_area_points = [
                point_color[0] for point_color in highlight_area_points_and_colors
            ]
            for point in highlight_area_points:
                color_map[point] = "red"
        plot_master = PlotMaster(r.dataset, data["labels"], r.order, color_map)
        reduced_plot = plot_input_data_reduced(value, plot_master)
        error_message = None

    return reduced_plot, error_message


"""
User controlled other plots
"""


@app.callback(
    Output("reduced-graph-custom", "figure"),
    Output("error-message-dim-custom", "children"),
    Input("plot_dropdown", "value"),
    Input("dendrogram-memory", "data"),
    Input("ClusterRadio", "value"),
    Input("custom-dendrogram-color-map", "data")
)
def plot_data_reduced_custom(value, data, highlight_area, colormap):
    if value is None:
        reduced_plot = go.Figure()
        error_message = (
            """Please select dimensionality reduction.\n 
            Ash will attempt to access pre-calculated dimensionality reduction data at the designated location (ash/common/user_data/reduced_dimensions).
            If the data is not found, 
            Ash will perform the necessary calculations on the matrix data provided at the designated location (ash/common/user_data/data.csv), 
            which may result in longer processing time."""
        )
    else:
        custom_color_map = copy.deepcopy(colormap)
        plot_master = PlotMaster(r.dataset, data["labels"], r.order, custom_color_map)
        reduced_plot = plot_input_data_reduced(value, plot_master)
        error_message = None

    return reduced_plot, error_message


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_silence_routes_logging=False)
