import ast
import re

from common.plot_master import PlotMaster


def plot_input_data_reduced(plot_input_data: str, plot_master: PlotMaster, color_map):
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
        return plot_master.plot_pca(color_map)
    elif "PCA_3D" in plot_input_data:
        return plot_master.plot_pca(color_map, dimensions=3)
    elif plot_input_data == "tSNE":
        return plot_master.plot_tsne(color_map)
    elif plot_input_data == "tSNE_3D":
        return plot_master.plot_tsne(color_map, dimensions=3)
    elif plot_input_data == "UMAP":
        return plot_master.plot_umap(color_map)
    elif plot_input_data == "UMAP_3D":
        return plot_master.plot_umap(color_map, dimensions=3)
    else:
        return plot_master.plot_pca()


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


def indexable_cycle(source, index):
    """
    Takes an iterable an and index
    if index is greater than the length of iterable

    """
    try:
        return source[index]
    except IndexError:
        return source[index % len(source)]


def modify_dendrogram_color(dendrogram, xmin, xmax, ymin, ymax, color):
    for i in range(len(dendrogram["icoord"])):
        if (
            xmin <= dendrogram["icoord"][i][0] <= xmax
            or xmin <= dendrogram["icoord"][i][2] <= xmax
        ) and (ymin <= dendrogram["dcoord"][i][1] <= ymax):
            dendrogram["color_list"][i] = color
            dendrogram["leaves_color_list"][i] = color


def write_to_text_file(filename, content):
    try:
        with open(filename, "w") as file:
            file.write(f"{content}")
        print(f"Content successfully written to {filename}")
    except IOError as e:
        print(f"Error writing to {filename}: {str(e)}")


def read_text_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
        print(f"Content successfully read from {filename}")
        return eval(content)
    except IOError as e:
        print(f"Error reading from {filename}: {str(e)}")
        return {}


def replace_color_values(dendrogram, color_map):
    for i, color in enumerate(dendrogram["color_list"]):
        for old_color, new_color in color_map:
            if color == old_color:
                dendrogram["color_list"][i] = new_color
                break
    return dendrogram


def get_click_coordinates(trace, points, selector):
    if points.point_inds:
        x = points.xs[0]
        y = points.ys[0]
        print(f"Clicked on point at (x={x}, y={y})")


def parse_value_string(value_string):
    list_string = re.search(r"\[(.*?)\]", value_string).group(1)
    value_list = ast.literal_eval(list_string)
    return value_list