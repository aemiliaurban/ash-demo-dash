import copy
from collections import Counter
from unittest.mock import patch

import matplotlib.pyplot
import pandas as pd
import plotly.graph_objects as go
from common.custom_threshold_plotly_dendrogram import create_dendrogram_modified
from common.data_parser import RDataParser
from common.plot_master import PlotMaster
from common.util import (
    assign_clusters,
    convert_to_dict,
    plot_input_data_reduced,
    write_to_text_file,
)
from dash import Dash, Input, Output, State, dash_table, dcc, html, ctx, get_asset_url
from layout import create_layout

matplotlib.pyplot.switch_backend("agg")

r = RDataParser()
r.parse()

app = Dash(__name__)
app.layout = create_layout(r)
server = app.server


@app.callback(
    Output("monocrit-list", "data"),
    [
        Input("monocrit-list", "data"),
        Input("split-button", "n_clicks"),
        Input("unsplit-button", "n_clicks"),
        Input("reset-button", "n_clicks"),
        State("split_point", "value"),
    ],
)
def add_split_point(
    monocrit_split_points,
    split_button_clicks,
    unsplit_button_clicks,
    reset_button_clicks,
    split_point_value,
):
    if not monocrit_split_points:
        monocrit_split_points = []
    if split_button_clicks > 0 and split_point_value:
        if ctx.triggered_id == "split-button":
            monocrit_split_points.append(int(split_point_value))
        elif ctx.triggered_id == "unsplit-button":
            monocrit_split_points = [
                point for point in monocrit_split_points if point != split_point_value
            ]
        elif ctx.triggered_id == "reset-button":
            monocrit_split_points = []
    return monocrit_split_points


@app.callback(
    Output("dendrogram-custom", "figure"),
    [Input("dendrogram-memory", "data")],
)
def plot_dendrogram(data):
    fig = go.Figure(data=data["data"], layout=data["layout"])
    return fig


@app.callback(
    Output("implicit_split_warning", "children"),
    [Input("monocrit-list", "data"), Input("dendrogram-memory", "data")],
)
def implicit_split_warning(monocrit_list, dendrogram_data):
    if len(set(dendrogram_data["cluster_indices"])) != len(monocrit_list) + 1:
        return html.Div(
            [
                html.Img(src=get_asset_url("warn_symbol.png"), style={'display': 'inline-block', 'height': '100px'}),
                html.Div(
                    "Current split points are not explicit. Please refer the Insrucions tab for more information.",
                    style={'display': 'inline-block', 'height': '100px'}
                ),
            ]
        )


@app.callback(
    Output("dendrogram-memory", "data"),
    Input("colorblind-palette-dropdown", "value"),
    Input("monocrit-list", "data"),
)
def create_dendrogram(colorblind_palette_input, monocrit_list):
    """
    Dendrogram initialization
    """
    colorblind_palette = (
        True if colorblind_palette_input == "Colorblind palette on" else False
    )
    # We are patching official plotly dendrogram to allow truncated tree
    # https://stackoverflow.com/questions/70801281/how-can-i-plot-a-truncated-dendrogram-plot-using-plotly
    # but the truncation part is actually not used, so maybe, we are not taking any kwargs in modified dendrogram
    # Hypothesis 2: we are doing it to allow for colorblind palette
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        custom_dendrogram = create_dendrogram(
            r.merge_matrix,
            labels=r.labels,
            colorblind_palette=colorblind_palette,
            monocrit_list=monocrit_list,
        )
        assigned_clusters = convert_to_dict(
            assign_clusters(custom_dendrogram.leaves_color_map_translated)
        )
        write_to_text_file("custom_dendrogram.txt", custom_dendrogram.dendro)

        return {
            "dendro": custom_dendrogram.dendro,
            "leaves_color_map_translated": custom_dendrogram.leaves_color_map_translated,
            "clusters": custom_dendrogram.clusters,
            "labels": custom_dendrogram.labels,
            "data": custom_dendrogram.data,
            "layout": custom_dendrogram.layout,
            "icoord": custom_dendrogram.xvals,
            "dcoord": custom_dendrogram.yvals,
            "assigned_clusters": assigned_clusters,
            "monocrit_list": monocrit_list,
            "cluster_indices": custom_dendrogram.cluster_indices,
            "color_map": custom_dendrogram.color_map,
        }


@app.callback(
    Output("no-of-clusters-output", "children"), Input("dendrogram-memory", "data")
)
def get_number_of_clusters(data):
    return f"Number of clusters: {data['clusters']}"


@app.callback(Output("clusters", "children"), Input("dendrogram-memory", "data"))
def get_cluster_percentages(data):
    return f"Individual cluster percentages: {Counter(data['cluster_indices'])}"


@app.callback(
    Output(component_id="cluster-stats-table", component_property="children"),
    [Input("dendrogram-memory", "data")],
)
def update_cluster_stats_table(data):
    counts_per_cluster = Counter(data["cluster_indices"])
    table_data = [
        {
            "Cluster ID": i,
            "Number of Samples": counts_per_cluster[i],
            "Share of Samples": round(
                counts_per_cluster[i] / sum(counts_per_cluster.values()), 2
            ),
            "Cluster Colour": "",
        }
        for i in sorted(counts_per_cluster)
    ]
    table = dash_table.DataTable(
        id="stats_table_content",
        data=table_data,
        row_selectable="single",
        selected_rows=[0],
        style_data_conditional=[
            {
                "if": {"row_index": i, "column_id": "Cluster Colour"},
                "background-color": data["color_map"][str(i + 1)],
            }
            for i in range(len(counts_per_cluster))
        ],
    )
    return html.Div(id="cluster-stats-table", children=table)


@app.callback(
    Output("heatmap-graph", "figure"),
    Input("dropdown-heatmap-plot", "value"),
    Input("dendrogram-memory", "data"),
    Input("stats_table_content", "selected_rows"),
    Input("colorblind-palette-dropdown", "value"),
)
def plot_heatmap(value, data, selected_rows: list[int], colorblind_palette_input):
    plot_master = PlotMaster(
        r.dataset, data["labels"], r.order, data["leaves_color_map_translated"]
    )
    # we use radio buttons, therefore only one option can be selected
    try:
        unfolded_cluster_nr = selected_rows[0] + 1
        mask = [i == unfolded_cluster_nr for i in data["cluster_indices"]]
        data_subset = r.dataset.loc[mask, :].reset_index(drop=True)

        colorblind_palette = (
            True if colorblind_palette_input == "Colorblind palette on" else False
        )

        if colorblind_palette:
            colorscale = "GnBu"
        else:
            colorscale = None

        fig = go.Figure(
            data=go.Heatmap(
                plot_master.df_to_plotly(data_subset, value),
                colorscale=colorscale,
                colorbar={"title": "Feature Value"},
            )
        )
        fig.update_layout({"xaxis_title": "Observation Number"})
        # empty name prevents trace name from being displayed in the hover tooltip
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "Observation Number: %{x}",
                    "Feature Name: %{y}",
                ]
            ),
            name="",
        )
        return fig
    except:
        # nothing selected, no need to handle, returning empty figure
        return go.Figure()


@app.callback(
    Output("two-features", "figure"),
    Output("error-message", "children"),
    Input("dropdown-selected-features-plot-1", "value"),
    Input("dropdown-selected-features-plot-2", "value"),
    Input("dendrogram-memory", "data"),
    prevent_initial_call=True,
)
def plot_two_selected_features(f1, f2, data):
    try:
        color_mask = [data["color_map"][str(i)] for i in data["cluster_indices"]]
        features = [f1, f2]
        if f1 == f2 and f1 is not None and f2 is not None:
            feature_plot = go.Figure()
            error_message = html.Div(
                [
                    html.Img(src=get_asset_url("warn_symbol.png"), style={'display': 'inline-block', 'height': '100px'}),
                    html.Div(
                        "Please select two different features.",
                        style={'display': 'inline-block', 'height': '100px'}
                    ),
                ]
            )
        else:
            color_map = copy.deepcopy(data["leaves_color_map_translated"])
            plot_master = PlotMaster(r.dataset, data["labels"], r.order, color_map)
            feature_plot = plot_master.plot_selected_features(features, color_mask)
            feature_plot.update_layout({"xaxis_title": f1, "yaxis_title": f2})
            feature_plot.update_traces(
                hovertemplate="<br>".join(
                    [
                        f"{f1}: %{{x}}",
                        f"{f2}: %{{y}}",
                    ]
                ),
                name="",
            )
            error_message = None
    except Exception:
        # Empty result
        feature_plot = go.Figure()
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
        color_mask = [data["color_map"][str(i)] for i in data["cluster_indices"]]
        plot_master = PlotMaster(r.dataset, data["labels"], r.order, color_map)
        reduced_plot = plot_input_data_reduced(value, plot_master, color_mask)
        error_message = None

    return reduced_plot, error_message


@app.callback(
    Output("download-data", "data"),
    State("dendrogram-memory", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_file(data, n_clicks):
    if n_clicks != 0:
        return dcc.send_data_frame(
            pd.concat(
                [
                    r.dataset,
                    pd.DataFrame({"ASSIGNED_CLUSTER": data["cluster_indices"]}),
                ],
                axis=1,
            ).to_csv,
            "mydf.csv",
        )


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True, dev_tools_silence_routes_logging=False)