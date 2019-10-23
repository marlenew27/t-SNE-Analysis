# -*- coding: utf-8 -*-

# Author: Tongjie Wang, Yihan Wang
# Date: Oct 9, 2019

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.manifold import TSNE
from typing import Union, List, Tuple, Optional, Callable, Any
from collections import namedtuple


DataPointInfo = namedtuple('DataPointInfo', ('sample_name', 'group_name', 'color'))
GroupInfo = namedtuple('GroupInfo', ('group_name', 'start', 'end', 'color'))


def _generate_save_file_name(path: str, postfix: str) -> str:
    """Small helper to insert postfix between file name and extension"""
    return '.{:}.'.format(postfix).join(path.rsplit('.', 1))


def read_data(file: str) -> pd.DataFrame:
    """
    Read given .xlsx or .csv file.

    :param file: .xlsx or .csv file
    :return: corresponding pandas.DataFrame object
    """
    if file.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.endswith('.csv'):
        return pd.read_csv(file)
    else:
        raise ValueError('unsupported file type (supported: .xlsx and .csv)')


def excel_col2num(col: str) -> int:
    """
    Convert Excel style letter column index to number

    :param col: str representing the column index in Excel style
    :return: corresponding column index in number (starting at zero)
    """
    num = 0
    for c in col:
        num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1


def get_column_labels(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None) -> List[str]:
    """
    Get the label of each column from a pandas.DataFrame object

    :param df: pandas.DataFrame object
    :param start: (optional) starting column (inclusive)
    :param end: (optional) ending column (exclusive)
    :return: list of str containing the labels of columns
    """
    return df.columns.tolist()[start:end]


def data_frame2matrix(df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None) -> np.ndarray:
    """
    Convert pandas.DataFrame into np.ndarray

    :param df: pandas.DataFrame object
    :param start: (optional) starting column (inclusive)
    :param end: (optional) ending column (exclusive)
    :return: corresponding np.ndarray matrix
    """
    return df.iloc[:, start:end].to_numpy()


def normalize_features(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize the given matrix

    :param matrix: np.ndarray matrix
    :return: normalized matrix
    """
    return scipy.stats.zscore(matrix)


def transform_tsne_model_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Train a 2-dimension t-SNE model with given matrix,
        whose rows are samples and columns are features

    :param matrix: a np.ndarray matrix
    :param n_comp: an int number of dimensions transformed
    :return: trained t-SNE model
    """
    return TSNE(n_components=2).fit_transform(matrix)

def transform_tsne_model_3d(matrix: np.ndarray) -> np.ndarray:
    """
    Train a 3-dimension t-SNE model with given matrix,
        whose rows are samples and columns are features

    :param matrix: a np.ndarray matrix
    :param n_comp: an int number of dimensions transformed
    :return: trained t-SNE model
    """
    return TSNE(n_components=3).fit_transform(matrix)

def get_data_points_info(groups_info: List[GroupInfo], column_labels: List[str]) -> List[DataPointInfo]:
    """
    Get DataPointInfo objects for all individual data points

    :param groups_info: list of GroupInfo objects
    :param column_labels: the column labels of the data
    :return: list of DataPointInfo objects
    """
    groups_col_intervals = list(map(lambda group_info: (
        group_info.start, group_info.end), groups_info))
    groups_col_labels = [
        column_labels[slice(*interval)] for interval in groups_col_intervals]
    data_points_info = []
    for group_labels, group_info in zip(groups_col_labels, groups_info):
        data_points_info.extend([DataPointInfo(
            label, group_info.group_name, group_info.color
        ) for label in group_labels])
    return data_points_info


# def cluster_snn(matrix: np.ndarray, max_pc=5, k=30) -> np.ndarray:
#     """
#     Alternative method to set up group colors: Use knn algorithm to cluster the similar features automatically
#
#     :param matrix: normalized matrix (zscore(matrix))
#     :param k: 30 (can be adjusted to get the best fit model; but the bigger k the more the model is overfitted)
#     :return: np.ndarray containing cluster info
#     """
#     import igraph as ig
#     from sklearn.neighbors import NearestNeighbors
#
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(matrix)
#     neighbor_graph = nbrs.kneighbors_graph(matrix)
#     g = ig.Graph()
#     g = ig.GraphBase.Adjacency(neighbor_graph.toarray().tolist(), mode=ig.ADJ_UNDIRECTED)
#     sim = np.array(g.similarity_jaccard())
#     g = ig.GraphBase.Weighted_Adjacency(sim.tolist(), mode=ig.ADJ_UNDIRECTED)
#
#     return np.array(g.community_multilevel(weights="weight", return_levels=False))


# Plotting Data Method 1: parse group info (annotation) from json param file:
def plot_tsne_matrix_2d(
    transformed: np.ndarray,
    data_points_info: List[DataPointInfo],
    figure_save_path: Optional[str] = '',
    show_fig: Optional[bool] = True
) -> None:
    """
    Plot first two t-SNEs of the transformed matrix data

    :param transformed: transformed data
    :param data_points_info: list of DataPointInfo for individual data point
    :param figure_save_path: path for saving the figure as a file
    :param show_fig: set to True to preview img in matplotlib
    """

    plt.figure(figsize=(8.0, 8.0))
    plt.tick_params(labelsize=20)

    for trans, info in zip(transformed, data_points_info):
        plt.plot(trans[0], trans[1], 'o', color=info.color)
        plt.annotate(info.sample_name, (trans[0], trans[1]))

    plt.xlabel('t-SNE 1', fontsize=20)
    plt.ylabel('t-SNE 2', fontsize=20)

    if figure_save_path:
        plt.savefig(_generate_save_file_name(figure_save_path, 'tsne_2d'))

    if show_fig:
        plt.show()


# Plotting Data Method 2: Use clustering as color groups
# def plot_tsne_cluster_2d(clu: np.ndarray, transformed: np.ndarray):
#     plt.scatter(transformed[:, 0], transformed[:, 1], c=clu, cmap=plt.colormaps().brg, lw=0, vmin=0, vmax=2)


def plot_tsne_matrix_3d(
    transformed: np.ndarray,
    data_points_info: List[DataPointInfo],
    figure_save_path: Optional[str] = '',
    show_fig: Optional[bool] = True,
    show_coordinates: Optional[bool] = False
) -> None:
    """
    Plot first three t-SNEs of the transformed data

    :param transformed: transformed data
    :param data_points_info: list of DataPointInfo for individual data point
    :param figure_save_path: path for saving the figure as a file
    :param show_fig: set to True to preview img in matplotlib
    :param show_coordinates: set True to show coordinates in figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8.0, 8.0))
    ax = fig.add_subplot(111, projection='3d')

    for trans, info in zip(transformed, data_points_info):
        ax.scatter(trans[0], trans[1], trans[2], c=info.color)
        ax.text(trans[0], trans[1], trans[2], '{} '.format(info.sample_name) + \
                ('({:.2f}, {:.2f}, {:.2f})'.format(trans[0], trans[1], trans[2]) if show_coordinates else ''))

    ax.set_xlabel('t-SNE 1', fontsize=20)
    ax.set_ylabel('t-SNE 2', fontsize=20)
    ax.set_zlabel('t-SNE 3', fontsize=20)

    if figure_save_path:
        plt.savefig(_generate_save_file_name(figure_save_path, 'tsne_3d'))

    if show_fig:
        plt.show()


'''
def plot_tsne_model_correlation(
        transformed: np.ndarray,
        correlations: pd.DataFrame,

)
    from sklearn.linear_model import LinearRegression
'''


def save_tsne_transformed(
    transformed: np.ndarray,
    tsne_count: int,
    data_points_info: List[DataPointInfo],
    csv_save_path: Optional[str] = '',
) -> None:
    """
    Export transformed matrix points' coordination into csv

    :param transformed: transformed data
    :param tsne_count: show first `tsne_count` components
    :param data_points_info: list of DataPointInfo for individual data point
    :param csv_save_path: path for saving the coordinations as csv file
    """
    rows = [['Sample Name', *('t-SNE {} '.format(i + 1) for i in range(tsne_count))]]
    rows.extend([info.sample_name, *(trans[i] for i in range(tsne_count))] \
                for trans, info in zip(transformed, data_points_info))
    with open(_generate_save_file_name(csv_save_path, 'tsne_coordinates'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# def save_cluster_areas(
#         features_name: List[str],
#         clu: np.ndarray,
#         csv_save_path: Optional[str] = '',
# ) -> None:
#     """
#     Export transformed matrix points' coordinate into csv
#
#     :param features_name: a column vector containing the names of all features
#     :param clu: clustering algorithm "cluster_snn"
#     :param csv_save_path: path for saving the coordinates as csv file
#     """
#     rows = [['Area', 'Cluster']]
#     rows.extend([])
#     cluster_areas = pd.DataFrame(data=np.array((features_name, clu)).T,columns=['Area', 'Cluster'])
#     cluster_areas.to_csv(_generate_save_file_name(csv_save_path, 'tsne_clusters'))
