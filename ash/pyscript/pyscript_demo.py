from __future__ import absolute_import

import math
from unittest.mock import patch


class RDataParser:
    def __init__(self, input_data):
        self.merge_matrix = [map(float, x) for x in input_data["merge_matrix"]]
        self.joining_height = [float(x) for x in input_data["joining_height"]]
        self.order = [float(x) for x in input_data["order"]]
        self.labels = input_data["labels"]
        self.max_tree_height: int = math.ceil(max(self.joining_height))
        self.height_marks: dict = self.create_height_marks()

    def convert_merge_matrix(self):
        transformed_matrix = []
        for node in self.merge_matrix:
            new_node = []
            for el in node:
                if el < 0:
                    transformed_el = abs(el) - 1
                else:
                    transformed_el = el + len(self.merge_matrix)
                new_node.append(transformed_el)
            transformed_matrix.append(new_node)

        self.merge_matrix = transformed_matrix

    def add_joining_height(self):
        # TODO: error for len merge matrix != len joining height
        for index in range(len(self.merge_matrix)):
            self.merge_matrix[index].append(self.joining_height[index])
            self.merge_matrix[index].append(self.order[index])

    def create_height_marks(self) -> dict[int | float, str]:
        height_marks = {}
        for step in range(len(self.joining_height)):
            height_marks[self.joining_height[step]] = f"Formed cluster {str(step+1)}"
        return height_marks


import os

import pandas as pd

INPUT_DATA_DENDROGRAM = {
    "merge_matrix": [
        [-1, -8],
        [-3, -5],
        [-6, -10],
        [-4, 3],
        [1, 4],
        [-2, 2],
        [-9, 6],
        [5, 7],
        [-7, 8],
    ],
    "joining_height": [
        282.45,
        537.97,
        629.66,
        844.635,
        1347.43916666667,
        1968.0425,
        3422.88555555556,
        6094.602725,
        19984.8865432099,
    ],
    "order": [6, 0, 7, 3, 5, 9, 8, 1, 2, 4],
    #    "order": [7, 1, 8, 4, 6, 1, 0, 9, 2, 3, 5],
    "labels": [
        "Alabama",  # 0
        "Alaska",  # 1
        "Arizona",  # 2
        "Arkansas",  # 3
        "California",  # 4
        "Colorado",  # 5
        "Connecticut",  # 6
        "Delaware",  # 7
        "Florida",  # 8
        "Georgia",  # 9
    ],
}

us_arrests = pd.read_csv(f"/Users/niki/diplomka/ash/ash/user_data/USArrests.csv")
us_arrests = us_arrests.head(10)
STATES = us_arrests["Unnamed: 0"]
US_ARRESTS = us_arrests.drop(["Unnamed: 0"], axis=1)

from unittest.mock import patch

import pandas as pd
import plotly.express as px
import streamlit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .plotly_modified_dendrogram import create_dendrogram_modified


class PlotMaster:
    def __init__(self, input_data, labels: list[str], order: list[int | float]):
        self.input_data = input_data
        self.labels = labels
        self.order = order

    def plot_interactive(self, func, data, threshold):
        return func(data, color_threshold=threshold, labels=self.labels)

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
        df = df.reindex(self.order)
        df = df[desired_columns].T
        return {"z": df.values, "x": self.order_labels(), "y": df.index.tolist()}

    def plot_pca(self):
        pca = PCA(2).fit_transform(self.input_data)
        fig = px.scatter(pca, x=0, y=1, color=self.labels)
        return fig

    def plot_pca_3d(self):
        pca = PCA(3).fit_transform(self.input_data)
        fig = px.scatter_3d(pca, x=0, y=1, z=2, color=self.labels)
        return fig

    def plot_all_dimensions(self):
        features = self.input_data.columns
        fig = px.scatter_matrix(self.input_data, dimensions=features, color=self.labels)
        fig.update_traces(diagonal_visible=True)
        return fig

    def plot_tsne(self):
        tsne = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(
            self.input_data
        )
        fig = px.scatter(tsne, x=0, y=1, color=self.labels, labels={"color": "states"})
        return fig

    def plot_tsne_3D(self):
        tsne = TSNE(n_components=3, random_state=0, perplexity=5).fit_transform(
            self.input_data
        )
        fig = px.scatter_3d(
            tsne, x=0, y=1, z=2, color=self.labels, labels={"color": "states"}
        )
        return fig

    def plot_umap(self):
        umap = UMAP(n_components=2, init="random", random_state=0).fit_transform(
            self.input_data
        )
        fig = px.scatter(umap, x=0, y=1, color=self.labels, labels={"color": "states"})
        return fig

    def plot_umap_3D(self):
        umap = UMAP(n_components=3, init="random", random_state=0).fit_transform(
            self.input_data
        )
        fig = px.scatter_3d(
            umap, x=0, y=1, z=2, color=self.labels, labels={"color": "states"}
        )
        return fig

    def plot_selected_features_streamlit(self):
        desired_columns = streamlit.multiselect(
            "Choose 2 features to plot.", self.input_data.columns
        )
        if len(desired_columns) != 2:
            streamlit.write("Please choose 2 features to plot.")
        else:
            to_plot = self.input_data[desired_columns]
            fig = px.scatter(
                to_plot,
                x="Murder",
                y="Assault",
                color=self.labels,
                labels={"color": "states"},
            )
            return fig

    def plot_custom_dendrogram(self, input_data, color_threshold):
        with patch(
            "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
            new=create_dendrogram_modified,
        ) as create_dendrogram:
            fig = self.plot_interactive(create_dendrogram, input_data, color_threshold)
            return fig


# -*- coding: utf-8 -*-

from collections import OrderedDict

from plotly import optional_imports
from plotly.graph_objs import graph_objs

# Optional imports, may be None for users that only use our core functionality.
np = optional_imports.get_module("numpy")
scp = optional_imports.get_module("scipy")
sch = optional_imports.get_module("scipy.cluster.hierarchy")
scs = optional_imports.get_module("scipy.spatial")


def create_dendrogram_modified(
    Z,
    orientation="bottom",
    labels=None,
    colorscale=None,
    hovertext=None,
    color_threshold=None,
):
    if not scp or not scs or not sch:
        raise ImportError(
            "FigureFactory.create_dendrogram requires scipy, \
                            scipy.spatial and scipy.hierarchy"
        )

    dendrogram = _Dendrogram_Modified(
        Z,
        orientation,
        labels,
        colorscale,
        hovertext=hovertext,
        color_threshold=color_threshold,
    )

    return graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)


class _Dendrogram_Modified(object):
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(
        self,
        X,
        orientation="bottom",
        labels=None,
        colorscale=None,
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
        hovertext=None,
        color_threshold=None,
    ):
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_dendrogram_traces(
            X, colorscale, hovertext, color_threshold
        )

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        """
        Returns colorscale used for dendrogram tree clusters.

        :param (list) colorscale: Colors to use for the plot in rgb format.
        :rtype (dict): A dict of default colors mapped to the user colorscale.

        """

        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        # This list is the colors that can be used by dendrogram, which were
        # determined as the combination of the default above_threshold_color and
        # the default color palette (see scipy/cluster/hierarchy.py)
        d = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            # TODO: 'w' doesn't seem to be in the default color
            # palette in scipy/cluster/hierarchy.py
            "w": "white",
        }
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            rgb_colorscale = [
                "rgb(0,116,217)",  # blue
                "rgb(35,205,205)",  # cyan
                "rgb(61,153,112)",  # green
                "rgb(40,35,35)",  # black
                "rgb(133,20,75)",  # magenta
                "rgb(255,65,54)",  # red
                "rgb(255,255,255)",  # white
                "rgb(255,220,0)",  # yellow
            ]
        else:
            rgb_colorscale = colorscale

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]

        # add support for cyclic format colors as introduced in scipy===1.5.0
        # before this, the colors were named 'r', 'b', 'y' etc., now they are
        # named 'C0', 'C1', etc. To keep the colors consistent regardless of the
        # scipy version, we try as much as possible to map the new colors to the
        # old colors
        # this mapping was found by inpecting scipy/cluster/hierarchy.py (see
        # comment above).
        new_old_color_map = [
            ("C0", "b"),
            ("C1", "g"),
            ("C2", "r"),
            ("C3", "c"),
            ("C4", "m"),
            ("C5", "y"),
            ("C6", "k"),
            ("C7", "g"),
            ("C8", "r"),
            ("C9", "c"),
        ]
        for nc, oc in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                # it could happen that the old color isn't found (if a custom
                # colorscale was specified), in this case we set it to an
                # arbitrary default.
                print("hello")
                print(KeyError)
                default_colors[n] = "rgb(0,116,217)"

        return default_colors

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": True,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for dendrogram figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_dendrogram_traces(self, Z, colorscale, hovertext, color_threshold):
        """
        Calculates all the elements needed for plotting a dendrogram.

        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
        :param (function) distfun: Function to compute the pairwise distance
                                   from the observations
        :param (function) linkagefun: Function to compute the linkage matrix
                                      from the pairwise distances
        :param (list) hovertext: List of hovertext for constituent traces of dendrogram
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves

        #"""
        P = sch.dendrogram(Z, color_threshold=color_threshold, labels=self.labels)

        icoord = scp.array(P["icoord"])
        dcoord = scp.array(P["dcoord"])
        ordered_labels = scp.array(P["ivl"])
        color_list = scp.array(P["color_list"])
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=colors[color_key]),
                text=hovertext_label,
                hoverinfo="text",
            )
            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, P["leaves"]


import json

import js
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px

r = RDataParser(INPUT_DATA_DENDROGRAM)
r.convert_merge_matrix()
r.add_joining_height()

plot_master = PlotMaster(US_ARRESTS, r.labels, r.order)


def create_dendrogram(value):
    with patch(
        "plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces",
        new=create_dendrogram_modified,
    ) as create_dendrogram:
        fig = plot_master.plot_interactive(create_dendrogram, r.merge_matrix, value)
    return fig.to_json()
