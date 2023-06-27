from ash.common.plot_master import PlotMaster


def plot_input_data_reduced(plot_input_data: str, plot_master: PlotMaster):
    """
    Plot the reduced input data based on the selected plot type.

    Args:
        plot_input_data (str): The selected plot type.
        plot_master (PlotMaster): An instance of the PlotMaster class.

    Returns:
        go.Figure: The plotted reduced input data.
    """
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


def extract_lowest_and_highest_x(data):
    """
    Extract the lowest and highest x-values from the data.

    Args:
        data: The input data. Assumes data in the Plotly dendrogram.data format
            and only takes points with 0 on the y axis.

    Returns:
        tuple: The lowest and highest x-values.
    """

    x_values = [point["x"] for point in data if 0 in point["y"]]
    return min(min(x_values)), max(max(x_values))


def assign_clusters(points):
    """
    Assign points to clusters based on their colors.

    Args:
        points: An ordered dictionary of points with their colors.

    Returns:
        list: A list of clusters where each cluster is a list of (point_id, color) tuples.
    """
    clusters = []
    current_cluster = []

    for point_id, color in points.items():
        if not current_cluster or current_cluster[-1][1] != color:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = []

        current_cluster.append((point_id, color))

    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def convert_to_dict(clusters):
    """
    Convert a list of clusters to a dictionary representation.

    Args:
        clusters: A list of clusters.

    Returns:
        dict: A dictionary where the keys are cluster labels and the values are lists of point IDs.
    """
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        point_ids = [point for point in cluster]
        cluster_dict[str(i)] = point_ids
    return cluster_dict


def calculate_cluster_percentages(data):
    """
    Calculate the percentage of points in each cluster.

    Args:
        data: A dictionary where the keys are cluster labels and the values are lists of point IDs.

    Returns:
        dict: A dictionary where the keys are cluster labels and the values are the percentage of points in each cluster.
    """
    total_length = 0

    for lst in data.values():
        total_length += len(lst)

    cluster_percentages = {}

    for i in range(len(list(data.values()))):
        cluster_percentages[f"{i}"] = round(
            (len(list(data.values())[i]) / total_length) * 100, 3
        )

    return cluster_percentages


def create_point_position_dictionary(lst: list[str]) -> dict[str, int]:
    """
    Calculate the size of each cluster.

    Args:
        data: A dictionary where the keys are cluster labels and the values are lists of point IDs.

    Returns:
        dict: A dictionary where the keys are cluster labels and the values are the sizes of each cluster.
    """
    dictionary = {}
    for index, item in enumerate(lst):
        dictionary[item] = index
    return dictionary


def get_elements_from_list(lst, positions):
    """
    Get the color associated with a cluster.

    Args:
        cluster: A list of (point_id, color) tuples.

    Returns:
        str: The color associated with the cluster.
    """
    try:
        marked_positions = []
        for position in positions:
            if 0 in lst[position]["y"]:
                marked_positions.append(lst[position])
        return marked_positions
    except IndexError:
        return []


def update_hover_template(hover_template, selected_points, highlight_area_points):
    updated_hover_template = hover_template

    if any(selected_points):
        selected_indices = [
            str(i) for i, selected in enumerate(selected_points) if selected
        ]
        selected_indices_str = ", ".join(selected_indices)
        highlight_area_str = ", ".join(highlight_area_points)
        updated_hover_template += (
            "<br><b>Selected Indices:</b> "
            + selected_indices_str
            + "<br><b>Highlight Area:</b> "
            + highlight_area_str
        )

    return updated_hover_template
