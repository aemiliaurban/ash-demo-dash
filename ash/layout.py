from dash import dash_table, dcc, html, get_asset_url


COMMON_STYLE = {"margin": "40px"}
COMMON_PADDING = {"padding-bottom": "10px"}


with open("./assets/instructions.md", "r") as file:
    INSTRUCTION_MD = file.read()

with open("./assets/about.md", "r") as file:
    ABOUT_MD = file.read()


def create_layout(data_parser):
    return html.Div(
        [
            # TODO: picture credits: https://artistcoveries.wordpress.com/2019/10/13/leaf-drawing-101/
            html.Img(src=get_asset_url("ash_logo.png"), style={"width": "100%"}),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Instructions",
                        children=[
                            html.Div(
                                [
                                    dcc.Markdown(
                                        INSTRUCTION_MD, dangerously_allow_html=True
                                    ),
                                ],
                                style=COMMON_STYLE,
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Interactive Clustering",
                        children=[
                            html.Div(
                                [
                                    dcc.Upload(
                                        children=html.Div(
                                            [
                                                html.Button(
                                                    "Upload",
                                                    id="Upload-button",
                                                    n_clicks=0,
                                                )
                                            ]
                                        ),
                                        multiple=True,
                                    ),
                                    html.H4("Enable colorblind palette"),
                                    dcc.Dropdown(
                                        [
                                            "Colorblind palette on",
                                            "Colorblind palette off",
                                        ],
                                        multi=False,
                                        id="colorblind-palette-dropdown",
                                        style=COMMON_PADDING,
                                    ),
                                    html.Br(),
                                    html.H1("Interactive Dendrogram"),
                                    dcc.Graph(
                                        id="dendrogram-custom",
                                    ),
                                    html.Div(id="implicit_split_warning"),
                                    html.H6("Split at Node:"),
                                    dcc.Input(id="split_point", type="number"),
                                    html.Button(
                                        "Add Split Point", id="split-button", n_clicks=0
                                    ),
                                    html.Button(
                                        "Remove Split Point",
                                        id="unsplit-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Remove All Split Points",
                                        id="reset-button",
                                        n_clicks=0,
                                    ),
                                    html.Br(),
                                    html.H1("Cluster Statistics"),
                                    html.Div(id="no-of-clusters-output"),
                                    html.Div(id="clusters"),
                                    html.Div(
                                        id="cluster-stats-table",
                                        children=dash_table.DataTable(
                                            id="stats_table_content"
                                        ),
                                    ),
                                    dcc.RadioItems(
                                        id="ClusterRadio", options=[], value=""
                                    ),
                                    html.Div(id="output-div"),
                                    html.Div(id="output-div-2"),
                                    html.Div(id="click-output", children=""),
                                    html.Div(id="min-output"),
                                    html.Div(id="max-output"),
                                    html.Div(id="output-div-manual"),
                                    html.H1("Heatmap"),
                                    dcc.Dropdown(
                                        list(data_parser.dataset.columns),
                                        multi=True,
                                        id="dropdown-heatmap-plot",
                                        value="All",
                                    ),
                                    dcc.Graph(
                                        id="heatmap-graph",
                                    ),
                                    html.Br(),
                                    html.H1("Two features plot"),
                                    html.Div(id="error-message"),
                                    dcc.Dropdown(
                                        list(data_parser.dataset.columns),
                                        multi=False,
                                        id="dropdown-selected-features-plot-1",
                                    ),
                                    dcc.Dropdown(
                                        list(data_parser.dataset.columns),
                                        multi=False,
                                        id="dropdown-selected-features-plot-2",
                                    ),
                                    dcc.Graph(
                                        id="two-features",
                                    ),
                                    html.Br(),
                                    html.H1(
                                        "Dimensionality reduction plot",
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
                                    dcc.Graph(
                                        id="reduced-graph",
                                    ),
                                    html.Div(id="error-message-dim-custom"),
                                    dcc.Store(id="dendrogram-memory"),
                                    dcc.Store(id="button-memory"),
                                    dcc.Store(id="monocrit-list"),
                                    dcc.Store(id="custom-dendrogram-color-map"),
                                    dcc.Store(id="click-values-output", data=[]),
                                    html.Div(
                                        [
                                            html.Br(),
                                            html.H1("Export Assigned Clusters to CSV"),
                                            html.Button(
                                                "Download File",
                                                id="save-button",
                                                n_clicks=0,
                                            ),
                                            html.Div(id="save-status"),
                                            dcc.Download(id="download-data"),
                                        ]
                                    ),
                                ],
                                style=COMMON_STYLE,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="About",
                        children=[
                            html.Div(
                                [
                                    dcc.Markdown(ABOUT_MD),
                                ],
                                style=COMMON_STYLE,
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )