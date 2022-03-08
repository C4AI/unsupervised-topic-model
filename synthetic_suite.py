#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters, make_blobs
from sklearn.cluster import SpectralCoclustering, KMeans
from sklearn.metrics import consensus_score, silhouette_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from sklearn.utils import  Bunch
from dataclasses import dataclass, field
from typing import Tuple
from queue import PriorityQueue
from pprint import pprint
import sys,os,time,re
import logging
from datetime import datetime
#from numba import jit, njit
from collections import deque, Counter
import pandas as pd

#TODO: vary k,l select best
#TODO: images for nbvd, wbkm

from my_utils import *
from nbvd import NBVD_coclustering
wbkm = __import__("wbkm numpy debugging land")
import algorithms

RNG_SEED=996535594000 # seed for reproducibility
#RNG_SEED=99654 # tested with 600,400, D- (3,5)
N_ROW_CLUSTERS, N_COL_CLUSTERS = 3,3 # number of row, column clusters
MAT_SHAPE = (600, 600) # matrix shape
SPARSITY = 0 # (Tasks 2,3) fill x % of matrix with zeroes
NOISE = 0 # (Tasks 2,3) fill x % of matrix with nasty noise


# B-CMatrix com 3,3 é fácil
# I- com 5,6 é interessante
DATASET_NAME = "B-CMatrix" # (Task 2) chosen dataset (regex expression)
ALG = 'nbvd' # (Task != 3) clustering algorithm
ATTEMPTS_MAX = 1 # (NBVD, WBKM) maximum attempts
SYMMETRIC = False # (NBVD) use symmetric NBVD algorithm?

TASK = 2 # 0: make_biclusters; 1: my weird gradient checkerboard; 2: single synthetic dataset; 3: all synthetic datasets
SHUFFLE_TEST = False # (Task 0) shuffle original matrix and use the clustering to try to recover it
MOVIE = False # (NBVD) display movie showing clustering iterations
WAIT_TIME = 4 # (Task 3) wait time between tasks
SHOW_IMAGES = True # (Task 3) display matrices
LABEL_CHECK = True
CENTROID_CHECK = True
COCLUSTER_CHECK = True
NORM_PLOT = True # (NBVD) display norm plot


RERUN_GENERATE = False # (Task 3) re-generate synthetic datasets
SYNTHETIC_FOLDER = 'synthetic/DataSets' # (Task 3) directory where synthetic datasets are stored
LOG_BASE_FOLDER = 'synthetic/logs' # (Task 3) base directory for logging and results sheet


def do_task_single (data, true_labels=None, only_one=True, alg=ALG, n_attempts=ATTEMPTS_MAX, 
        show_images=True, first_image_save_path=None, RNG_SEED=None, logger=None):
    RNG = np.random.default_rng(RNG_SEED)
    if logger:
        logger.info(f"shape: {data.shape}")
    else:
        print(f"shape: {data.shape}")
    timer = None if only_one else WAIT_TIME * show_images 

    if not only_one:
        # plot original data to build suspense AND save figure if a save path is provided
        plot_matrices([data], ["Original dataset"], timer=timer, savefig=first_image_save_path)

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
            verbose=True, save_history=MOVIE, save_norm_history=NORM_PLOT, logger=logger)
    elif alg == 'wbkm':
        model = wbkm.WBKM_coclustering(data, n_clusters=N_ROW_CLUSTERS, n_attempts=n_attempts,
            random_state=RNG_SEED, verbose=True, logger=logger)
    elif alg == 'spectral':
        model = SpectralCoclustering(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        model.fit(data)

    ### exclude
    elif alg == 'kmeans':
        model = lambda : None
        # does this make sense i dont know
        row_model = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        row_model.fit(data)
        model.row_labels_ = row_model.labels_
        col_model = KMeans(n_clusters=N_COL_CLUSTERS, random_state=RNG_SEED)
        col_model.fit(data.T)
        model.column_labels_ = col_model.labels_
    ### exclude

    elif ALG == 'nbvd_waldyr':
        U, S, V, resNEW, itr = algorithms.NBVD(data, N_ROW_CLUSTERS, N_COL_CLUSTERS, itrMAX=2000)
        model = lambda: None
        model.U, model.S, model.V = U, S, V
        model.biclusters_, model.row_labels_, model.column_labels_ = NBVD_coclustering.get_stuff(U, V.T)
        print("resNEW, itr:", resNEW, itr) # resNEW is norm squared; itr is iteration_no
    
    # show animation of clustering process
    if MOVIE and alg == 'nbvd':
        pyqtgraph_thing(data, model, 25)

    #########################
    # evaluate results 
    #########################

    ### external indices:
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

    ### internal indices
    # print silhouette scores
    silhouette = print_silhouette_score(data, model.row_labels_, model.column_labels_, logger=logger)
    
    ### DBG: testing different methods of labelling
    if alg == 'nbvd':
        _, row1, col1 = NBVD_coclustering.get_stuff(model.R, model.C, model.B, method="not fancy")
        #_, row2, col2 = NBVD_coclustering.get_stuff(model.R, model.C, model.B, method="fancy")
        _, row3, col3 = NBVD_coclustering.get_stuff(model.R, model.C, model.B, model.data, model.centroids, method="centroids")

        print("not fancy")
        silhouette2 = print_silhouette_score(data, row1, col1, logger=logger)
        #print("fancy")
        #silhouette2 = print_silhouette_score(data, row2, col2, logger=logger)
        print("centroids")
        silhouette3 = print_silhouette_score(data, row3, col3, logger=logger)
        print("rows:")
        thing = lambda arr : sorted(Counter(arr).items())
        print(thing(row1), 
            #thing(row2),
            thing(row3), sep="\n")
        #print(row1)
        #print(row2)
        #print(row3)
        print("cols:")
        print(thing(col1), 
            #thing(col2), 
            thing(col3), sep="\n")
    
    if show_images:
        # shade lines/columns of original dataset
        if LABEL_CHECK:
            shaded_label_matrix(data, model.row_labels_, kind="rows",method_name="rows", RNG=RNG)
            shaded_label_matrix(data, model.column_labels_, kind="columns",method_name="columns", RNG=RNG)
            if COCLUSTER_CHECK and alg == "nbvd":
                shade_coclusters(data, (model.row_labels_, model.column_labels_), 
                    model.cluster_assoc, RNG=RNG)
            
            
        # centroid (and dataset) (normalized) scatter plot
        if CENTROID_CHECK and alg == 'nbvd':
            row_centroids, col_centroids = model.centroids[0], model.centroids[1]
            centroid_scatter_plot(data, row_centroids, model.row_labels_, kind="row", RNG=RNG)
            centroid_scatter_plot(data.T, col_centroids, model.column_labels_, kind="col", RNG=RNG)
        # norm evolution
        if hasattr(model, "norm_history"):
            plot_norm_history(model)

        # general plots
        if alg == 'nbvd':
            to_plot = [data, model.R@model.B@model.C]
            names = ["Original dataset", "Reconstructed matrix RBC"]
        elif alg == 'wbkm':
            to_plot = [data, model.D1@model.P@model.S@model.Q.T@model.D2, model.P@model.S@model.Q.T]
            names = ["Original dataset", "Reconstructed matrix...?", "Matrix that looks funny sometimes"]
        elif alg =="nbvd_waldyr":
            to_plot = [data, model.U@model.S@model.V.T, model.S]
            names = ["Original dataset", "Reconstructed matrix USV.T", "Block value matrix S"]
        plot_matrices(to_plot, names, timer = None if (not SHUFFLE_TEST and only_one) else 2*timer)
    
        if SHUFFLE_TEST:
            # rearranged data
            fit_data = data[np.argsort(model.row_labels_)]
            fit_data = fit_data[:, np.argsort(model.column_labels_)]
            if show_images:
                plot_matrices([data, fit_data], ["Original dataset", "After biclustering (rearranged to show biclusters)"], timer=timer)

    # return general statistics
    if alg == 'nbvd':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            best_iter=model.best_iter, best_norm=model.best_norm, n_attempts=n_attempts)
    elif alg == 'wbkm':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            max_iter_reached=model.best_max_iter_reached, best_norm=model.best_norm,
            no_zero_cols=model.best_no_zero_cols, n_attempts=n_attempts)
    elif alg == 'spectral':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), n_attempts=1)
    elif alg == 'kmeans':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), n_attempts=1)

    return bunch

def fill_sheet (sheet, results, task_name):
    pretty = {
    'best_iter': 'Iterações',
    'best_norm': 'Norma (objetivo) final',
    'n_attempts': 'Tentativas',
    'silhouette': 'Silhouette score',
    'max_iter_reached': 'Iterações',
    'no_zero_cols': 'Nº clusters nulos',
    }
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
    RNG, RNG_SEED = start_default_rng(seed=RNG_SEED)
    np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)

    # prepare data
    if TASK==0:
        data, rows, columns = make_biclusters(
            shape=MAT_SHAPE, n_clusters=N_ROW_CLUSTERS, shuffle=False, random_state=RNG_SEED,
            noise=0, # no negative values
            minval=0.3,
            maxval=300
        )
    elif TASK==1:
        data = random_block_matrix(*MAT_SHAPE, n_row_clusters=N_ROW_CLUSTERS, n_col_clusters=N_COL_CLUSTERS, seed=RNG_SEED)
    elif TASK==2 or TASK==3:
        # generate synthetic datasets if necessary
        datasets = make_synthetic_datasets(MAT_SHAPE, NOISE/100, SPARSITY, SYNTHETIC_FOLDER, 
            seed=RNG_SEED, rerun_generate=RERUN_GENERATE, print_info=(TASK==2))
        if TASK == 2:
            for name in datasets:
                if re.search(DATASET_NAME, name):
                    task_name = name
                    data = datasets[name]
                    break # stop at first match

    # do the actual task
    if TASK == 0:
        do_task_single(data, true_labels=(rows, columns), RNG_SEED=RNG_SEED)
    elif TASK == 1 or TASK == 2:
        if TASK == 2:
            s1,s2=cool_header_thing(), cool_header_thing()
            print(f"""{s1}  {''.join(list(reversed(s1)))}\n{s2} {''.join(list(reversed(s2)))}
            Task: {task_name}\n{s1} {''.join(list(reversed(s1)))}\n{s2}  {''.join(list(reversed(s2)))}""")
        do_task_single(data, only_one=True, alg=ALG, RNG_SEED=RNG_SEED)
    elif TASK == 3:
        n_attempts = ATTEMPTS_MAX
        alg_list = ['nbvd', 'wbkm', 'spectral', 'kmeans']
        #alg_list = ['wbkm', 'spectral'] 

        # setup logging and results sheet
        os.makedirs(LOG_BASE_FOLDER, exist_ok=True)
        now = datetime.today().strftime(r'%Y-%m-%d__%H-%M-%S')
        log_folder = os.path.join(LOG_BASE_FOLDER, now)
        logger = logger_setup(log_folder)
        sheet_path = os.path.join(log_folder, '___results.xlsx')
        #writer = pd.ExcelWriter(path=sheet_path) # openpyxl by default
        writer = pd.ExcelWriter(path=sheet_path, engine='xlsxwriter')

        # dump globals
        file_handler_add(logger, "Ω_globals")
        dump_globals(globals(), logger=logger)

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
                        show_images=SHOW_IMAGES, first_image_save_path=first_image_save_path, 
                        RNG_SEED=RNG_SEED, logger=logger)
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
            sheet_name = f"{alg.upper()}, n_clusters=({N_ROW_CLUSTERS},{N_COL_CLUSTERS})"
            sheet.to_excel(writer, sheet_name=sheet_name)
            adjust_column_width(sheet, writer, sheet_name=sheet_name)

            # get best and worst results
            log_best_worst(pq, name=alg.upper(), logger=logger)

        writer.save() # else it won't save
        
if __name__ == "__main__":
    main()
