#!/usr/bin/env python3 

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.datasets import make_biclusters, make_blobs
from sklearn.cluster import SpectralCoclustering, KMeans
from sklearn.decomposition import *
from sklearn.preprocessing import *
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
        sl = difs[i-window//2 : i+window//2+extra]
        new[i] = sum(sl) / len(sl)
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

def plot_matrices(matrices : Iterable, names : Iterable, 
                    timer=None, savefig : str = None, aspect_ratio=1):
    """
    Plot the given matrices using matplotlib (in different plots) using
    the given names as titles.
    Optionally wait 'timer' seconds. (Set to None to block execution.)
    Also optionally, save figure to savefig. (No exception handling is performed.)
    Note that 'timer=0' results in no plot being shown, 
    but the figure being saved if savefig is given."""
    for X, name in zip(matrices, names):
        plt.matshow(X, cmap=plt.cm.Blues, aspect=aspect_ratio)
        plt.title(name)
    if timer != 0:
        plt.show(block = not timer)
    if savefig:
        plt.savefig(savefig)
    if type(timer) == int:
        if timer != 0:
            plt.pause(timer)
        plt.close(fig='all')

def shaded_label_matrix (data, labels, kind, method_name=None, RNG=None, opacity=0.09, aspect_ratio=1):
    RNG = RNG or np.random.default_rng()
    fig, ax = plt.subplots()
    ax.matshow(data, cmap=plt.cm.Blues)

    # colors for shading
    n = 1+max(labels) # 0-indexed
    palette = RNG.choice(list(mcolors.CSS4_COLORS.values()), size=n, replace=False) # XKCD_COLORS ?
    colors = [palette[label] for label in labels]
    legend_dict = {}

    xs, ys = np.arange(data.shape[1]), np.arange(data.shape[0])
    view = data if kind == "rows" else data.T
    for i, row in enumerate(view):
        if kind == "rows":
            ax.fill_between(xs, i, i+1, color=colors[i], alpha=opacity)
        else:
            ax.fill_betweenx(ys, i, i+1, color=colors[i], alpha=opacity)
        
        # keep track of legends
        if labels[i] not in legend_dict:
            legend_artist = mpatches.Patch(color=colors[i], label=labels[i])
            legend_dict[labels[i]] = legend_artist
    # show plot
    ax.set_aspect(aspect_ratio)
    plt.title(f"Shaded dataset ({kind}): {method_name if method_name else ''}")
    labels, handles = list(zip(*legend_dict.items()))
    plt.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="upper left")

def shade_coclusters (data, row_col_labels : Tuple[int], cluster_assoc, RNG=None, aspect_ratio=1):
    RNG = RNG or np.random.default_rng()
    fig, ax = plt.subplots()
    row_labels, col_labels = row_col_labels
    ax.matshow(data, cmap=plt.cm.Blues, alpha=0) # just to orient axes

    # colors for shading
    n = cluster_assoc.sum() # significant (True) associations
    palette = RNG.choice(list(mcolors.CSS4_COLORS.values()), size=n, replace=False) # XKCD_COLORS ?
    color_dict, max_color = {}, -1
    legend_dict = {}

    xs, ys = np.arange(data.shape[1] + 1), np.arange(data.shape[0])
    for y in ys:
        to_fills = {} # list of regions to be painted

        for x in xs[:-1]:
            possible_cocluster = (row_labels[y], col_labels[x])
            is_cocluster = cluster_assoc[possible_cocluster]
            if is_cocluster:
                # keep track of colors and legends
                if possible_cocluster not in color_dict:
                    max_color += 1 # new color
                    color_dict[possible_cocluster] = palette[max_color]
                    legend_artist = mpatches.Patch(color=color_dict[possible_cocluster], label=possible_cocluster)
                    legend_dict[possible_cocluster] = legend_artist
                
                color = color_dict[possible_cocluster]
                if color not in to_fills:
                    to_fills[color] = np.full(len(xs), False) # array indicating regions to be painted

                # paint cocluster (later)
                to_fill = to_fills[color]
                to_fill[x:x+2] = True # 2+ True values specify an x region to be filled
        
        # fill_between cannot handle a color array apparently and will just pick the first color [matplotlib 3.4.3]
        for color, to_fill in to_fills.items():
            ax.fill_between(xs, y, y+1, where=to_fill, color=color) #RNG.random((len(xs),4))
    # show plot
    ax.set_aspect(aspect_ratio)
    plt.title(f"Shaded coclusters")
    labels, handles = list(zip(*legend_dict.items()))
    plt.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="upper left")

def get_centroids_by_cluster (data, labels, n_clusters):
    n_smp, n_dim = data.shape
    centroids = np.zeros((n_dim, n_clusters), dtype=np.float64)

    for k in range(n_clusters):
        mask = (labels == k)
        centroids[:, k] = np.mean(data[mask, :], axis=0)
    return centroids

def centroid_scatter_plot (samples, centroids, labels, title, pca=None, palette=None, RNG=None):
    """Dimension-reduced plot of samples and centroids, coloring points according to labels.
    """
    RNG = RNG or np.random.default_rng()
    # seed = RNG.integers(0, 2147483647) #DBG: better colors not doing an additional call to RNG
    # TODO: kind should be just a title string

    _, n = centroids.shape
    points = normalize(np.vstack([samples, centroids.T]), axis=1) # normalize before PCA
    #pca = TruncatedSVD(n_components=2)
    #pca = KernelPCA(n_components=2)
    if not pca:
        pca = PCA(n_components=2, random_state=42)
        pca.fit(points)
    reduced_points = pca.transform(points)
    #print("samples, features:", pca_row.n_samples_, pca_row.n_features_)
    #print("reduced_points:", reduced_points.shape)
    palette = palette if (not palette is None) else RNG.choice(list(mcolors.CSS4_COLORS.values()), size=n, replace=False) # XKCD_COLORS ?
    colors = [palette[label] for label in labels]

    fig = plt.figure()
    ax = fig.add_subplot(1, 21, (1,18)) # make a subplot that spans the left-side (n-1)/n of the figure.
    
    # plot centroids in a way that allows us to label them
    things = []
    for i in range(n):
        # note: a[-2:-1] returns the 2nd to last value; a[-1:0] does not; so, we do a[-1:][0]
        thing = ax.scatter(reduced_points[-n+i: , 0][0], reduced_points[-n+i: , 1][0], color=palette[i], marker="s", s=400, alpha=0.8)
        things.append(thing)
    ax.scatter(reduced_points[:-n , 0], reduced_points[:-n , 1], color=colors)
    legend = ax.legend(things, list(range(n)), bbox_to_anchor=(0.99,1), loc="upper left")
    plt.gca().add_artist(legend) # manually add legend so we can add new legends later
    plt.title(title)
    return (pca, palette, ax)

def plot_norm_history (model):
    y = model.norm_history
    fig, ax = plt.subplots()
    ax.plot(range(len(y)), y)
    vel = get_difs(y, 10)
    acc = get_difs(vel, 5)

    acc = np.array(acc)
    # TODO: test if ok
    #args = np.arange(len(acc))[acc > acc.mean()]
    #fig.vlines(args, ymin=min(y)-2*abs(min(y)), ymax=max(y)+abs(max(y)), color="black", alpha=0.2)
    #print(args)
    for i,val in enumerate(acc):
        if val > sum(acc)/len(acc):
            ax.axvline(i, color="black", alpha=0.1)
    plt.title("Norm history")

def adjust_column_width (dataframe, writer, sheet_name):
    # handle width of index
    first_col_width = max([len(v) for v in dataframe.index.astype(str)])
    writer.sheets[sheet_name].set_column(0, 0, first_col_width) # set width of a range of columns
    
    # handle width of columns
    for col_idx, column_name in enumerate(dataframe.columns, start=1): # offset 1 to account for 'task name' column
        col_width = max(len(column_name), max([len(v) for v in dataframe[column_name].astype(str)]))
        writer.sheets[sheet_name].set_column(col_idx, col_idx, col_width) # set width of a range of columns

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

def pyqtgraph_thing (data, model, ms_period):
    app, win, im, anchor = init_qt_graphics(data)
    my_update = lambda : update_display(model.best_history, im, anchor)
    time = QtCore.QTimer()
    time.timeout.connect(my_update)
    time.start(ms_period)
    win.show()
    app.exec_() # start qt app
