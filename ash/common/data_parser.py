import math
import os

import pandas as pd

DATA_FOLDER = os.path.join(os.getcwd(), "common", "user_data")


def csv_order_data_reader(path: str):
    data = []
    with open(path, newline="") as f:
        for line in f.readlines():
            if line != '"x"\n':
                data.append(float(line) - 1)
    return data


def csv_merge_data_reader(path: str):
    data = []
    with open(path, newline="") as f:
        for line in f.readlines():
            if line != '"V1","V2"\n':
                data.append([int(x) for x in line.split(",")])
    return data


class RDataParser:
    def __init__(self):
        self.dataset = self.read_dataset()
        self.merge_matrix = [map(float, x) for x in self.read_merge_matrix()]
        self.joining_height = [float(x) for x in self.read_joining_height()]
        self.order = [float(x) for x in self.read_order_data()]
        self.labels = self.read_labels()
        self.input_flow_data_dendrogram = {
            "merge_matrix": self.merge_matrix,
            "joining_height": self.joining_height,
            "order": self.order,
            "labels": self.labels,
        }
        self.max_tree_height: int = math.ceil(max(self.joining_height))
        self.height_marks: dict = self.create_height_marks()

    @staticmethod
    def read_dataset():
        return pd.read_csv(os.path.join(DATA_FOLDER, "data.csv"))

    @staticmethod
    def read_order_data():
        order_raw = pd.read_csv(os.path.join(DATA_FOLDER, "order.csv"))[
            "x"
        ].values.tolist()
        order = [x - 1 for x in order_raw]
        return order

    @staticmethod
    def read_joining_height():
        return pd.read_csv(os.path.join(DATA_FOLDER, "heights.csv"))[
            "x"
        ].values.tolist()

    @staticmethod
    def read_merge_matrix():
        merge_matrix_raw = pd.read_csv(os.path.join(DATA_FOLDER, "merge.csv"))
        merge_matrix_V1 = merge_matrix_raw["V1"].values.tolist()
        merge_matrix_V2 = merge_matrix_raw["V2"].values.tolist()
        merge_matrix = [list(x) for x in zip(merge_matrix_V1, merge_matrix_V2)]
        return merge_matrix

    def read_labels(self):
        try:
            labels = pd.read_csv(os.path.join(DATA_FOLDER, "labels.csv"))
        except:
            labels = [i for i in range(len(self.order))]
        return labels

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
