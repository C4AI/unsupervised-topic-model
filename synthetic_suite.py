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
from typing import Tuple
from queue import PriorityQueue
from pprint import pprint
import sys,os,time
import logging
from datetime import datetime
from numba import jit, njit
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl
import pyqtgraph as pg
from collections import deque 
import pandas as pd

from my_utils import MeanTuple, cool_header_thing, plot_matrices, start_default_rng, print_silhouette_score, bic_boolean_to_labels, pyqtgraph_thing
from nbvd import NBVD_coclustering
wbkm = __import__("wbkm numpy debugging land")
from generateSyntheticData_mod import generateSyntheticData

RNG_SEED=996535595 # seed for reproducibility
N_ROW_CLUSTERS, N_COL_CLUSTERS = 3,3 # number of row, column clusters
MAT_SIZE = 600 # matrix size (square)

ALG = 'nbvd' # (Task != 3) clustering algorithm
ATTEMPTS_MAX = 10 # (NBVD, WBKM) maximum attempts
SYMMETRIC = False # (NBVD) use symmetric NBVD algorithm?
TASK = 3 # 0: make_biclusters; 1: my weird gradient checkerboard; 2: single synthetic dataset; 3: all synthetic datasets
SHUFFLE_TEST = False # (Task 0) shuffle original matrix and use the clustering to try to recover it

MOVIE = False # (NBVD) display movie showing clustering iterations
SYNTHETIC_DATASET = 22 # (Task 2) chosen dataset
WAIT_TIME = 4 # (Task 3) wait time between tasks
SHOW_IMAGES = False # (Task 3) display matrices
RERUN_GENERATE = False # (Task 3) re-generate synthetic datasets
SYNTHETIC_FOLDER = 'synthetic/DataSets' # (Task 3) directory where synthetic datasets are stored
LOG_BASE_FOLDER = 'synthetic/logs' # (Task 3) base directory for logging and results sheet

np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)


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

def get_synthetic_data_list (data_dir):
    return os.listdir(data_dir)

def load_dataset (data_dir, filename):
    path = os.path.join(data_dir, filename)
    return np.loadtxt(path, delimiter=';')

def do_task_single (data, true_labels=None, only_one=True, alg=ALG, n_attempts=ATTEMPTS_MAX, 
        show_images=True, first_image_save_path=None, logger=logging.getLogger(__name__)):
    logger.info(f"shape: {data.shape}")
    timer = None if only_one else WAIT_TIME * show_images 

    if not only_one:
        plot_matrices([data], ["Original dataset"], timer=timer, savefig=first_image_save_path) # plot original data to build suspense

    if SHUFFLE_TEST:
        # shuffle clusters
        row_idx = RNG.permutation(data.shape[0])
        col_idx = RNG.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]
        if show_images:
            plot_matrices([data], ["Shuffled dataset"], timer=timer)

    # do co-clustering
    if alg == 'nbvd':
        model = NBVD_coclustering(data, symmetric=SYMMETRIC, n_row_clusters=N_ROW_CLUSTERS, 
            n_col_clusters=N_COL_CLUSTERS, n_attempts=n_attempts, random_state=RNG_SEED, 
            verbose=True, save_history=False, logger=logger)
    elif alg == 'wbkm':
        model = wbkm.WBKM_coclustering(data, n_clusters=N_ROW_CLUSTERS, n_attempts=n_attempts,
            random_state=RNG_SEED, verbose=True, logger=logger)
    elif alg == 'spectral':
        model = SpectralCoclustering(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        model.fit(data)
    elif alg == 'kmeans':
        model = lambda : None
        # does this make sense i dont know
        row_model = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        row_model.fit(data)
        model.row_labels_ = row_model.labels_
        col_model = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        col_model.fit(data.T)
        model.column_labels_ = col_model.labels_
    
    if MOVIE and alg == 'nbvd':
        pyqtgraph_thing(data, model)

    #########################
    # evaluate results 
    #########################
    # external indices:
    if TASK==0:
        rows, columns = true_labels
        if SHUFFLE_TEST:
            bic_true = (rows[:, row_idx], columns[:, col_idx])
        else:
            bic_true = (rows, columns)
        bic_pred = model.biclusters_
        con_score = consensus_score(bic_true, bic_pred)
        logger.info(f"\nconsensus score: {con_score:.3f}\n")

        # retrieve integer labels
        pred_rows, pred_columns = model.row_labels_, model.column_labels_
        true_rows, true_columns = bic_boolean_to_labels(bic_true)

        # rows (samples):
        ari = adjusted_rand_score(true_rows, pred_rows)
        ami = adjusted_mutual_info_score(true_rows, pred_rows)
        vmeasure = v_measure_score(true_rows, pred_rows)
        logger.info(f"rows:\n  ARI= {ari:.3f}\n  AMI= {ami:.3f}\n  VMs= {vmeasure:.3f}\n")

        # columns (features/attributes):
        ari = adjusted_rand_score(true_columns, pred_columns)
        vmeasure = v_measure_score(true_columns, pred_columns)
        ami = adjusted_mutual_info_score(true_columns, pred_columns)
        logger.info(f"columns:\n  ARI= {ari:.3f}\n  AMI= {ami:.3f}\n  VMs= {vmeasure:.3f}\n")

    # internal indices
    silhouette = print_silhouette_score(data, model.row_labels_, model.column_labels_)

    # matplotlib
    if alg == 'nbvd':
        to_plot = [data, model.R@model.B@model.C, model.B]
        names = ["Original dataset", "Block value matrix RBC", "Block value matrix B"]
        if show_images:
            plot_matrices(to_plot, names, timer = None if (not SHUFFLE_TEST and only_one) else 2*timer)

    if SHUFFLE_TEST:
        # rearranged data
        fit_data = data[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        if show_images:
            plot_matrices([data, fit_data], ["Original dataset", "After biclustering; rearranged to show biclusters"], timer=timer)

    if alg == 'nbvd':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            best_iter=model.best_iter, best_norm=model.best_norm, n_attempts=n_attempts)
    elif alg == 'wbkm':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            max_iter_reached=model.best_max_iter_reached,
            no_zero_cols=model.best_no_zero_cols, n_attempts=n_attempts)
    elif alg == 'spectral':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), n_attempts=n_attempts)
    elif alg == 'kmeans':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), n_attempts=n_attempts)
    return bunch

pretty = {
    'best_iter': 'Iterações',
    'best_norm': 'Norma (objetivo) final',
    'n_attempts': 'Tentativas',
    'silhouette': 'Silhouette score',
    'max_iter_reached': 'Iterações',
    'no_zero_cols': 'Nº clusters nulos',
}
def fill_sheet (sheet, results, task_name):
    series = pd.Series(name=task_name, dtype=object) # series name will be the row name
    if results:
        for name, value in results.items():
            series[pretty[name]] = value
    return sheet.append(series)

def get_worst5 (pq : PriorityQueue):
    things = []
    while not pq.empty() and len(things) < 5:
        things.append(pq.get())
    return things

def get_best5 (pq : PriorityQueue):
    things = []
    while not pq.empty():
        thing = pq.get()
        qsize = pq.qsize() # probably ok?
        if qsize <= 5:
            things.append(thing)
    return things

def log_best_worst (pq : PriorityQueue, name, logger):
    worst5 = get_worst5(pq)
    best5 = get_best5(pq)
    file_handler_add(logger, f'__{name}_best_and_worst')

    # worst
    s1,s2=cool_header_thing(), cool_header_thing()
    logger.info(f"\n\n{s1} WORST 5: {s1}\n")
    for mt_silhouette, task_name in worst5:
        logger.info(f"{task_name}: {mt_silhouette}")
    logger.info(f"\n{s1}#########{s1}\n")

    # best
    logger.info(f"\n{s2} BEST 5: {s2}\n")
    for mt_silhouette, task_name in best5:
        logger.info(f"{task_name}: {mt_silhouette}")
    logger.info(f"\n{s2}#########{s2}")

def main():
    global RNG_SEED
    RNG = start_default_rng(seed=RNG_SEED)
    RNG_SEED = RNG_SEED if RNG_SEED else RNG.integers(low=0, high=2**30)

    if TASK==0:
        data, rows, columns = make_biclusters(
            shape=(MAT_SIZE, MAT_SIZE), n_clusters=N_ROW_CLUSTERS, shuffle=False, random_state=RNG_SEED,
            noise=0, # no negative values
            minval=0.3,
            maxval=300
        )
    elif TASK==1:
        data = random_block_matrix((MAT_SIZE, MAT_SIZE), n_row_clusters=N_ROW_CLUSTERS, n_col_clusters=N_COL_CLUSTERS, seed=RNG_SEED)
    elif TASK==2 or TASK==3:
        # generate synthetic datasets fi necessary
        os.makedirs(SYNTHETIC_FOLDER, exist_ok=True)
        if RERUN_GENERATE or not os.listdir(SYNTHETIC_FOLDER):
            # empty folder if not already empty
            for f in os.listdir(SYNTHETIC_FOLDER):
                os.remove(os.path.join(SYNTHETIC_FOLDER, f))
            generateSyntheticData(MAT_SIZE, MAT_SIZE, 0, directory=SYNTHETIC_FOLDER)

        synthetic_data_names = get_synthetic_data_list(SYNTHETIC_FOLDER)
        datasets = []
        for i,filename in enumerate(synthetic_data_names):
            dataset = load_dataset(SYNTHETIC_FOLDER, filename)
            task_name = filename.replace(".txt", "")
            if dataset.ndim == 2: # only matrices
                datasets.append((dataset, task_name))
        datasets.sort(key = lambda t : t[1]) # sort datasets by task_name

        # print info about synthetic tasks
        if TASK==2:       
            print("Datasets available:")
            print(*[name for d,name in datasets], sep="\n")
            data, task_name = datasets[SYNTHETIC_DATASET] # 5,22 ok # but for n_clusters=5,5?
            s1,s2=cool_header_thing(), cool_header_thing()
            print(f"""\n{s1}  {''.join(list(reversed(s1)))}\n{s2} {''.join(list(reversed(s2)))}
            Task: {task_name}\n{s1} {''.join(list(reversed(s1)))}\n{s2}  {''.join(list(reversed(s2)))}""")

    if TASK == 0:
        do_task_single(data, true_labels=(rows, columns))
    elif TASK == 1 or TASK == 2:
        do_task_single(data)
    elif TASK == 3:
        n_attempts = ATTEMPTS_MAX
        alg_list = ['nbvd', 'wbkm', 'spectral', 'kmeans'] 

        # setup logging and results sheet
        os.makedirs(LOG_BASE_FOLDER, exist_ok=True)
        now = datetime.today().strftime(r'%Y-%m-%d__%H-%M-%S')
        log_folder = os.path.join(LOG_BASE_FOLDER, now)
        logger = logger_setup(log_folder)
        sheet_path = os.path.join(log_folder, '___results.xlsx')
        writer = pd.ExcelWriter(path=sheet_path)

        # do several experiments
        for alg in alg_list:
            sheet = pd.DataFrame() # create new empty sheet for this alg
            pq = PriorityQueue()
            for d, task_name in datasets:
                file_handler_add(logger, f"{task_name}_{alg.upper()}")
                s1,s2=cool_header_thing(), cool_header_thing()
                logger.info(f"""{s1}  {''.join(list(reversed(s1)))}\n{s2} {''.join(list(reversed(s2)))}
                Task: {task_name}\n{s1} {''.join(list(reversed(s1)))}\n{s2}  {''.join(list(reversed(s2)))}""")
                image_path = os.path.join(log_folder, f"{task_name}.png")
                first_image_save_path = image_path if not os.path.exists(image_path) else None # save one figure for each task

                try:
                    results = do_task_single(d, only_one=False, alg=alg, n_attempts=n_attempts, 
                        show_images=SHOW_IMAGES, first_image_save_path=first_image_save_path)
                except Exception as e:
                    results = None # blank line
                    logger.info(str(e)) # show exception text but don't stop
                    if SHOW_IMAGES:
                        time.sleep(3)
                else:
                    mt_silhouette = results.silhouette
                    pq.put((mt_silhouette, task_name))
                finally:
                    sheet = fill_sheet(sheet, results, task_name=task_name)
            sheet.to_excel(writer, sheet_name=f"{alg.upper()}, n_clusters=({N_ROW_CLUSTERS},{N_COL_CLUSTERS})")
            writer.save() # else it won't save

            # get best and worst results
            log_best_worst(pq, name=alg.upper(), logger=logger)
        
if __name__ == "__main__":
    main()
