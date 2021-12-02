#!/usr/bin/env python3 

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters, make_blobs
from sklearn.cluster import SpectralCoclustering, KMeans
from sklearn.metrics import consensus_score, silhouette_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from sklearn.utils import Bunch
from dataclasses import dataclass, field
from typing import Tuple, Iterable, Union
from queue import PriorityQueue
from pprint import pprint
import sys,os,time
import logging
from datetime import datetime
from numba import jit, njit
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl
import pyqtgraph as pg
from collections import deque, OrderedDict
import pandas as pd

from generateSyntheticData_mod import generateSyntheticData

############################################################################## 
# to use a set number of cpus: 
#   taskset --cpu-list 0-7 python "synthetic_suite.py"
##############################################################################

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

def make_synthetic_datasets (mat_shape, noise_prob : float, sparsity : Union[int, list], 
    synthetic_folder, seed=None, rerun_generate=False, print_info=False) -> OrderedDict :
    """Write datasets to disk (if necessary); optionally print info about datasets 
    
    Returns the loaded datasets.
    """
    RNG = np.random.default_rng(seed)

    def load_dataset (data_dir, filename):
        path = os.path.join(data_dir, filename)
        return np.loadtxt(path, delimiter=';')

    datasets = OrderedDict()
    sparsity = sparsity if type(sparsity) == list else [sparsity]
    for sp in sparsity:
        # write datasets to disk (if necessary)
        synthetic_folder_shape_sparsity = os.path.join(synthetic_folder, f"{mat_shape[0]}-{mat_shape[1]}-{sp}")
        os.makedirs(synthetic_folder_shape_sparsity, exist_ok=True)
        if rerun_generate or not os.listdir(synthetic_folder_shape_sparsity):
            print("Generating synthetic datasets ...")
            # empty folder if not already empty
            for f in os.listdir(synthetic_folder_shape_sparsity):
                os.remove(os.path.join(synthetic_folder_shape_sparsity, f))
            generateSyntheticData(*mat_shape, sp, directory=synthetic_folder_shape_sparsity, seed=seed)

        # load datasets from disk
        synthetic_data_names = os.listdir(synthetic_folder_shape_sparsity)
        for i,filename in enumerate(sorted(synthetic_data_names)):
            dataset = load_dataset(synthetic_folder_shape_sparsity, filename)
            task_name = filename.replace(".txt", "")
            if dataset.ndim == 2 and task_name not in datasets: # only matrices
                dataset = noisify(dataset, probability=noise_prob, RNG=RNG)
                datasets[task_name] = dataset

    # print info about synthetic tasks
    if print_info:       
        print("Datasets available:")
        print(*[name for name in datasets], sep="\n", end="\n\n")
    return datasets

def plot_matrices(matrices : Iterable, names : Iterable, 
                    timer=None, savefig : str = None):
    """
    Plot the given matrices using matplotlib (in different plots) using
    the given names as titles.
    Optionally wait 'timer' seconds. (Set to None to block execution.)
    Also optionally, save figure to savefig. (No exception handling is performed.)
    Note that 'timer=0' results in no plot being shown, 
    but the figure being saved if savefig is given."""
    for X, name in zip(matrices, names):
        plt.matshow(X, cmap=plt.cm.Blues)
        plt.title(name)
    if timer != 0:
        plt.show(block = not timer)
    if savefig:
        plt.savefig(savefig)
    if type(timer) == int:
        if timer != 0:
            plt.pause(timer)
        plt.close(fig='all')

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

def dump_globals (globals_dict, logger=None):
    cool_types = {int, tuple, str, bool, list}
    banned_keys = {'__name__', '__file__'}
    log_or_print = logger.info if logger else print
    for k, v in globals_dict.items():
        if k not in banned_keys and type(v) in cool_types:
            log_or_print(f"{k:^15}:\t\t{v}")

def noisify (sparse_matrix, probability, RNG=None):
    """Noisifies matrix in-place and returns it."""
    RNG = RNG or np.random.default_rng()
    flat_length = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    min_val, max_val = sparse_matrix.min(), sparse_matrix.max()

    np.place(sparse_matrix, 
        sparse_matrix == 0, 
        (RNG.random(size=(flat_length,)) < probability) * RNG.uniform(min_val, max_val, size=(flat_length,))
    )
    return sparse_matrix

def get_difs (a, window):
    
    new, difs = a.copy(), a.copy()
    n = len(a)
    extra = window%2
    for i,v in enumerate(a[:-1]):
        difs[i] = a[i+1] - a[i]
    difs[n-1] = difs[n-2]

    for i, v in enumerate(a):
        if i < window//2 or i > n - window//2:
            continue
        new[i] = sum(difs[i-window//2 : i+window//2+extra]) / window
    for i in range(window//2):
        new[i] = new[window//2]
        new[n-i-1] = new[n-window//2]
    return new
        

def cool_header_thing ():
    print_rng = default_rng()
    n1 = print_rng.integers(low=14, high=21)
    n2 = print_rng.integers(low=14, high=21)
    s = "".join(print_rng.permutation(list(n1* "#" + n2* "@")))
    return s

def mat_debug (M: np.ndarray, name=None, entries=None, logger=None):
    name = name if name else "matrix"
    s = cool_header_thing()
    header =  f"\n\n{s}  {name}  {s}\n"
    if logger:
        logger.info(header)
        logger.info(f"dtype, shape, min, max, norm: {M.dtype} | {M.shape}\n{np.abs(M).min()} | {np.abs(M).max()} | {np.linalg.norm(M)}")
    else:
        print(header)
        print(f"dtype, shape, min, max, norm: {M.dtype} | {M.shape}\n{np.abs(M).min()} | {np.abs(M).max()} | {np.linalg.norm(M)}")
    if entries is True or (M.shape[0] * M.shape[1] < 2000 and entries is None):
        pprint(M)

def count_zero_columns (M : np.ndarray):
    M_col_sum = np.sum(M, axis=0)
    zero_cols = np.argwhere( (M_col_sum == 0))
    return zero_cols.shape[0]

#def print_silhouette_score (data, row_labels, column_labels, logger=logging.getLogger(__name__)):
def print_silhouette_score (data, row_labels, column_labels, logger=None):
    sil_score_row = silhouette_score(data, row_labels)
    sil_score_col = silhouette_score(data.T, column_labels)
    results_message = f"\nsilhouette score:\n\trows: {sil_score_row:.3f}\n\tcols: {sil_score_col:.3f}\n"
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

# TODO: decide: DELETE?
def random_block_matrix (shape, n_row_clusters, n_col_clusters, seed=None):
    """more like 'some blocky gradient thing'"""

    rng = np.random.default_rng(seed=seed)
    k, l = n_row_clusters, n_col_clusters
    #block_map = rng.integers(size=(k,l), low=0, high=k*l//4+1)
    block_map = 2*np.reshape(np.arange(k*l), (k,l))
    
    block_list=[]
    elem_shape = (shape[0]//k, shape[1]//l)
    for i in range(k):
        block_line = []
        for j in range(l):
            elem = block_map[i,j] * np.ones(elem_shape)
            block_line.append(elem)
        block_list.append(block_line)
    block_matrix = np.block(block_list)
    #block_matrix = np.reshape(block_matrix, shape) # doesnt fix the finnickiness

    return block_matrix

def init_qt_graphics(data_matrix):
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget()
    win.resize(1400,800)
    pg.setConfigOption('imageAxisOrder', 'row-major') # NOTE: data is by default in column-major order
    font=QtGui.QFont()
    font_size=45
    font.setPixelSize(font_size)
    posy, posx = data_matrix.shape
    actualpos = 0.49*posx - 0.1*font_size, -0.10*posy
    text_color = QtGui.QColor(61, 95, 101)
    grad = pg.GradientEditorItem()
    #available LUTs: ‘thermal’, ‘flame’, ‘yellowy’, ‘bipolar’, ‘spectrum’, ‘cyclic’, ‘greyclip’, ‘grey’, ‘viridis’, ‘inferno’, ‘plasma’, ‘magma’
    # i like spectrum, grey, cyclic, bipolar
    grad.loadPreset('cyclic')
    
    im1,p1,anchor1 = pg.ImageItem(), pg.PlotItem(), pg.TextItem()
    #p.setWindowTitle('Original matrix') # no work
    p1.invertY(True)
    im1.setImage(data_matrix)
    im1.setLookupTable(grad.getLookupTable(512))
    anchor1.setText("") # fake text just so positioning is the same for both graphics views
    anchor1.setColor(QtGui.QColor(61, 95, 101))
    anchor1.setFont(font)
    p1.addItem(anchor1)
    p1.addItem(im1)
    win.addItem(p1)
    anchor1.setPos(*actualpos) # lazy centering
    
    im2,p2,anchor2 = pg.ImageItem(), pg.PlotItem(), pg.TextItem()
    #p2.setWindowTitle('Reconstructed matrix')
    p2.invertY(True)
    im2.setLookupTable(grad.getLookupTable(512))
    anchor2.setText("0")
    anchor2.i, anchor2.timer = 0,0 # keep track of iteration number
    anchor2.setColor(text_color)
    anchor2.setFont(font)
    anchor2.setPos(*actualpos)
    #print(anchor2.x(), anchor2.y())
    p2.addItem(anchor2)
    p2.addItem(im2)
    win.addItem(p2)
    
    return app, win, im2, anchor2

def qt_plot_matrix (data, win):
    # FIXME: DELETE dis
    # create a new subplot and plot a thing
    p = pg.PlotItem()
    im = pg.ImageItem(data)
    p.setWindowTitle('pyqtgraph!!')
    p.invertY(True)
    """ # looks bad
    grad = pg.GradientEditorItem()
    #available LUTs: ‘thermal’, ‘flame’, ‘yellowy’, ‘bipolar’, ‘spectrum’, ‘cyclic’, ‘greyclip’, ‘grey’, ‘viridis’, ‘inferno’, ‘plasma’, ‘magma’
    grad.loadPreset('flame')
    im.setLookupTable(grad.getLookupTable(512))
    """

    p.addItem(im)
    win.addItem(p)
    
def update_display(history : deque, display, anchor):
    if anchor.timer:
        anchor.timer -=1
    else:
        actual_history_len = len(history)
        #next_matrix = history.popleft()
        R,B,C = history.popleft()
        next_matrix = R@B@C
        display.setImage(next_matrix)
        anchor.setText(str(anchor.i))
        anchor.i = (anchor.i+1) % actual_history_len
        history.append((R,B,C)) # loop history!  
        if anchor.i == 0: # stop for a bit at the final one
            anchor.timer = len(history)//3

def pyqtgraph_thing (data, model):
    app, win, im, anchor = init_qt_graphics(data)
    my_update = lambda : update_display(model.best_history, im, anchor)
    time = QtCore.QTimer()
    time.timeout.connect(my_update)
    time.start(50)
    win.show()
    app.exec_() # start qt app

def adjust_column_width (dataframe, writer, sheet_name):
    # handle width of index
    first_col_width = max([len(v) for v in dataframe.index.astype(str)])
    writer.sheets[sheet_name].set_column(0, 0, first_col_width) # set width of a range of columns
    
    # handle width of columns
    for col_idx, column_name in enumerate(dataframe.columns, start=1): # offset 1 to account for 'task name' column
        col_width = max(len(column_name), max([len(v) for v in dataframe[column_name].astype(str)]))
        writer.sheets[sheet_name].set_column(col_idx, col_idx, col_width) # set width of a range of columns
