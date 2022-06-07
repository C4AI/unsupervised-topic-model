#!/usr/bin/env python3 

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import *
from sklearn.metrics import silhouette_score
from sklearn.utils import Bunch
from dataclasses import dataclass, field
from typing import Tuple, Iterable, Union
import os
import logging
from collections import deque, OrderedDict

@dataclass(order=True)
class MeanTuple:
    """Compare float tuples using their mean."""
    numbers: Tuple[float] = field(compare=False)
    mean: float = field(init=False)
    
    def __init__(self, *args):
        self.numbers = args
        self.mean = sum(self.numbers) / len(self.numbers)
    
    def __str__(self):
        template = ", ".join(["{:.3f}" for _ in self.numbers])
        return "(" + template.format(*self.numbers) + ")"

def start_default_rng (seed=None, logger=None):
    """ Start the default RNG with a (user-provided / randomly-generated) seed 
    so we don't have to store the RNG's starting state (which is less human-friendly).
    
    Returns an RNG and a seed (same as the one given if not None).
    """

    if seed:
        new_seed = seed
    else:
        temp_rng = default_rng()
        new_seed = temp_rng.integers(low=0, high=2**30) # get a seed so we don't have to store
    new_rng = default_rng(new_seed)
    if logger:
        logger.info(f"Random seed is: {new_seed}\n")
    else:
        print(f"Random seed is: {new_seed}\n")
    return (new_rng, new_seed)

def logger_setup(log_folder : str, level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(level) # handler levels take precedence over logger level

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter and add stuff to stuff
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter) # add formatter to ch
    logger.addHandler(ch) # add ch to logger

    # create log folder for this run
    os.makedirs(log_folder, exist_ok=True)
    logger.log_folder = log_folder # add attribute for sneaky access

    return logger

def file_handler_add(logger : logging.Logger, name : str, replace=True, level=logging.DEBUG):
    log_path = os.path.join(logger.log_folder, f'{name}.log')
    fh = logging.FileHandler(filename=log_path, encoding='utf-8')
    fh.setLevel(level)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    if replace and len(logger.handlers) == 2:
        logger.removeHandler(logger.handlers[-1])
    logger.addHandler(fh)

def dump_globals (globals_dict, logger=None):
    cool_types = {int, tuple, str, bool, list}
    banned_keys = {'__name__', '__file__'}
    log_or_print = logger.info if logger else print
    for k, v in globals_dict.items():
        if k not in banned_keys and type(v) in cool_types:
            log_or_print(f"{k:^15}:\t\t{v}")
        

def cool_header_thing ():
    print_rng = default_rng()
    n1 = print_rng.integers(low=14, high=21)
    n2 = print_rng.integers(low=14, high=21)
    s = "".join(print_rng.permutation(list(n1* "#" + n2* "@")))
    return s

def count_zero_columns (M : np.ndarray):
    M_col_sum = np.sum(M, axis=0)
    zero_cols = np.argwhere( (M_col_sum == 0))
    return zero_cols.shape[0]

#def print_silhouette_score (data, row_labels, column_labels, logger=logging.getLogger(__name__)):
def print_silhouette_score (data, row_labels, column_labels, logger=None):
    sil_score_row = silhouette_score(data, row_labels)
    sil_score_col = silhouette_score(data.T, column_labels)
    row_labels_bin, col_labels_bin = np.bincount(row_labels), np.bincount(column_labels)
    results_message = f"\nrow label count: {row_labels_bin}\ncol label count: {col_labels_bin}\nsilhouette score:\n\trows: {sil_score_row:.3f}\n\tcols: {sil_score_col:.3f}\n"
    if logger:
        logger.info(results_message)
    else:
        print(results_message)
    return (sil_score_row, sil_score_col)

def bic_boolean_to_labels (bic):
    rows, cols = bic
    labelize = lambda a: np.argmax(a, axis=0)
    row_labels, col_labels = labelize(rows), labelize(cols)
    return row_labels, col_labels

def get_centroids_by_cluster (data, labels, n_clusters):
    n_smp, n_dim = data.shape
    centroids = np.zeros((n_dim, n_clusters), dtype=np.float64)

    for k in range(n_clusters):
        mask = (labels == k)
        centroids[:, k] = np.mean(data[mask, :], axis=0)
    return centroids


def adjust_column_width (dataframe, writer, sheet_name):
    # handle width of index
    first_col_width = max([len(v) for v in dataframe.index.astype(str)])
    writer.sheets[sheet_name].set_column(0, 0, first_col_width) # set width of a range of columns
    
    # handle width of columns
    for col_idx, column_name in enumerate(dataframe.columns, start=1): # offset 1 to account for 'task name' column
        col_width = max(len(column_name), max([len(v) for v in dataframe[column_name].astype(str)]))
        writer.sheets[sheet_name].set_column(col_idx, col_idx, col_width) # set width of a range of columns
