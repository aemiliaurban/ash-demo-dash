from __future__ import absolute_import

from collections import OrderedDict

import numpy as np
import scipy
from common.color_mappings import (
    COLORBLIND_PALETTE,
    KELLY_MAX_CONTRAST_PALETTE,
)
from common.util import indexable_cycle


def split_dendrogram(linkage_matrix, monocrit, palette):
    """
    The function create scipy dendrogram object, so that
    link colors matches monocrit split criteria
    """
    dflt_col = "#808080"
    # 1 get clusters
    cluster_indices = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, 0, criterion="monocrit", monocrit=monocrit
    )
    # 2 create color map
    color_map = {
        cluster: indexable_cycle(palette, cluster) for cluster in set(cluster_indices)
    }
    point_color_map = {}
    for nr, p in enumerate(cluster_indices):
        point_color_map[nr] = color_map[p]
    # 3 apply color map
    link_cols = {}
    for i, i12 in enumerate(linkage_matrix[:, :2].astype(int)):
        c1, c2 = (
            link_cols[x] if x > len(linkage_matrix) else point_color_map[x] for x in i12
        )
        link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else dflt_col
    return (
        scipy.cluster.hierarchy.dendrogram(
            Z=linkage_matrix, link_color_func=lambda x: link_cols[x]
        ),
        cluster_indices,
        # numpy int is not serializable, therefore color_map cannot be simpy returned
        {int(k): v for k, v in color_map.items()},
    )


def sort_dendrogram(dendrogram):
    def sorting_key(i):
        return max(i[2])

    s = sorted(
        zip(
            dendrogram["color_list"],
            dendrogram["icoord"],
            dendrogram["dcoord"],
            dendrogram["ivl"],
            dendrogram["leaves"],
            dendrogram["leaves_color_list"],
        ),
        key=sorting_key,
    )

    return {
        "color_list": [i[0] for i in s],
        "icoord": [i[1] for i in s],
        "dcoord": [i[2] for i in s],
        "ivl": [i[3] for i in s],
        "leaves": [i[4] for i in s],
        "leaves_color_list": [i[5] for i in s],
    }


def create_dendrogram_modified(
    Z,
    orientation="bottom",
    labels=None,
    hovertext=None,
    colorblind_palette=False,
    monocrit_list=[],
):
    dendrogram = _Dendrogram_Modified(
        Z,
        orientation,
        labels,
        hovertext=hovertext,
        colorblind_palette=colorblind_palette,
        monocrit_list=monocrit_list,
    )
    return dendrogram


class _Dendrogram_Modified:
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(
        self,
        X,
        orientation="bottom",
        labels=None,
        xaxis="xaxis",
        yaxis="yaxis",
        hovertext=None,
        colorblind_palette=False,
        monocrit_list=[],
    ):
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}
        self.colorblind_palette = colorblind_palette
        if self.colorblind_palette:
            self.palette = COLORBLIND_PALETTE
        else:
            self.palette = KELLY_MAX_CONTRAST_PALETTE
        self.monocrit_list = monocrit_list

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        (
            dendro,
            dd_traces,
            xvals,
            yvals,
            ordered_labels,
            leaves,
            leaves_color_map_translated,
            clusters,
        ) = self.get_dendrogram_traces(X, hovertext)

        self.labels = ordered_labels
        self.leaves = leaves
        self.leaves_color_map_translated = leaves_color_map_translated
        self.clusters = clusters
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.xvals = xvals
        self.yvals = yvals

        self.dendro = dendro

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
        self.layout = self.set_figure_layout()
        self.data = dd_traces

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

    def set_figure_layout(self):
        """
        Sets and returns default layout object for dendrogram figure.
        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": True,
                "hovermode": "closest",
                "xaxis_title": "Observations",
                "yaxis_title": "Height",
                "xaxis_showticklabels": False,
            }
        )

        return self.layout

    def get_dendrogram_traces(self, Z, hovertext):
        """
        Calculates all the elements needed for plotting a dendrogram.

        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
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

        """
        # TODO: Z neni linkage matrix ale list of ...
        Z = np.asarray(Z)
        cut_points = np.zeros((Z.shape[0],))
        inverted_monocrit = [-i for i in self.monocrit_list]
        cut_points[inverted_monocrit] = 1

        # We need to sort links in order to color them properly
        dendrogram, cluster_indices, color_map = split_dendrogram(
            Z, cut_points, self.palette
        )
        self.cluster_indices = cluster_indices
        self.color_map = color_map
        P = sort_dendrogram(dendrogram)
        # that is simply an integer that denotes number of clusters
        clusters = len(set(cluster_indices))
        # icoord is list of x coordinates for each '∩' shape - that is 4 values, because '∩' has for vertices
        icoord = scipy.array(P["icoord"])
        # dcoord is list of y coordinates for each '∩' shape - that is 4 values, because '∩' has for vertices
        dcoord = scipy.array(P["dcoord"])

        ordered_labels = scipy.array(P["ivl"])
        color_list = list(P["color_list"])
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

            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=P["color_list"][i]),
                hoverinfo="skip",
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
            # append midpoint labels
            # ["blue" if (len(icoord) - j) in self.monocrit_list not in self.monocrit_list else P["color_list"][i]  for j in range(len(icoord))]
            # print(self.monocrit_list)
            # marker size
            markers = dict(color=P["color_list"][i])
            if (len(icoord) - i) in self.monocrit_list:
                markers = dict(color="red", size=15, symbol="x")
            trace_list.append(
                dict(
                    type="scatter",
                    x=[(xs[1] + xs[2]) / 2],
                    y=[(ys[1] + ys[2]) / 2],
                    mode="markers",
                    marker=markers,
                    text=hovertext_label,
                    hoverinfo="text",
                    hovertext=[f"Node {len(icoord) - i}"],
                )
            )

        leaves_color_list_translated = OrderedDict()

        # for i in range(len(P["leaves_color_list"])):
        #     leaves_color_list_translated[ordered_labels[i]] = colors[  # TODO: tady failujeme, potrebujeme to vubec?
        #         P["leaves_color_list"][i]
        #     ]

        return (
            P,
            trace_list,
            icoord,
            dcoord,
            ordered_labels,
            P["leaves"],
            leaves_color_list_translated,
            clusters,
        )