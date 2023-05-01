import streamlit
from common.data_parser import RDataParser
from common.plot_master import PlotMaster
from common.plotly_modified_dendrogram import create_dendrogram_modified
from unittest.mock import patch


def plot_input_data_reduced(plot_input_data: list[str]):
    if "Select two features" in plot_input_data:
        two_features_plot = plot_master.plot_selected_features_streamlit(col2)
        if two_features_plot:
            col2.plotly_chart(two_features_plot)
    if "All dimensions" in plot_input_data:
        all_dimensions = plot_master.plot_all_dimensions()
        col2.plotly_chart(all_dimensions)
    if "PCA" in plot_input_data:
        pca = plot_master.plot_pca()
        col2.plotly_chart(pca)
    if "PCA_3D" in plot_input_data:
        pca = plot_master.plot_pca(dimensions=3)
        col2.plotly_chart(pca)
    if "tSNE" in plot_input_data:
        tsne = plot_master.plot_tsne()
        col2.plotly_chart(tsne)
    if "tSNE_3D" in plot_input_data:
        tsne = plot_master.plot_tsne(dimensions=3)
        col2.plotly_chart(tsne)
    if "UMAP" in plot_input_data:
        umap = plot_master.plot_umap()
        col2.plotly_chart(umap)
    if "UMAP_3D" in plot_input_data:
        umap = plot_master.plot_umap(dimensions=3)
        col2.plotly_chart(umap)


r = RDataParser()
r.convert_merge_matrix()
r.add_joining_height()

streamlit.set_page_config(layout="wide")
col1, col2 = streamlit.columns(2)
col1.header("Cluster Analysis")

color_threshold = col1.slider("Select color threshold.", 0, r.max_tree_height)
desired_features = col1.multiselect(
    "What features would you like to plot in a heatmap.", list(r.dataset.columns)
)
colorblind_palette_choice = col1.selectbox("Use colorblind palette", ("Yes", "No"))
colorblind_palette = True if colorblind_palette_choice == "Yes" else False


def compute_dendrogram_fig():
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        custom_dendrogram = create_dendrogram(
            r.merge_matrix,
            color_threshold=color_threshold,
            colorblind_palette=colorblind_palette,
        )
        custom_dendrogram_color_map = custom_dendrogram.leaves_color_map_translated
    return custom_dendrogram, custom_dendrogram_color_map


custom_dendrogram, custom_dendrogram_color_map = compute_dendrogram_fig()


plot_master = PlotMaster(
    r.dataset, custom_dendrogram.labels, r.order, custom_dendrogram_color_map
)

dendrogram_figure = plot_master.plot_dendrogram(custom_dendrogram)

col1.plotly_chart(dendrogram_figure)


fig_heatmap = plot_master.plot_heatmap(desired_features)

col1.plotly_chart(fig_heatmap)

plot_input_data = col2.multiselect(
    "How would you like to plot the input data.",
    [
        "Select two features",
        "All dimensions",
        "PCA",
        "PCA_3D",
        "tSNE",
        "tSNE_3D",
        "UMAP",
        "UMAP_3D",
    ],
)

plot_input_data_reduced(plot_input_data)
