#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters, make_blobs
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score, silhouette_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score

from dataclasses import dataclass, field
from pprint import pprint
import sys,os
from numba import jit, njit
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl
import pyqtgraph as pg
from collections import deque
import logging

from my_utils import cool_header_thing, start_default_rng, mat_debug, count_zero_columns

# TODO: fix first exit condition

ITER_MAX = 2000 #2000
ATTEMPTS_MAX = 5

"""
SpectralCoclustering.biclusters_ is the tuple: (row_truths, col_truths), where row_truths is a boolean matrix
with NO_CLUSTERS lines indicating whether the jth row belongs to the ith cluster (and
col_truths is analogous).
Caution: data was shuffled using row_idx, so rows won't have the real labels.

SpectralCoclustering.row_labels_ is an N array (if data is N x M) 
with the integer (0-indexed) indices for each row. (And column_labels_ works analogously.)
"""
# DO: remember to flip P and Q so they have the correct shape

@dataclass(eq=False) # generated eq method might not be ideal?
class NBVD_coclustering:
    # init arguments
    data: np.ndarray
    n_row_clusters: int
    n_col_clusters: int
    symmetric: bool = False
    iter_max: int = ITER_MAX
    n_attempts: int = ATTEMPTS_MAX
    random_state: int = None
    verbose: bool = False
    save_history: bool = False
    logger : logging.Logger = None

    # properties calculated post init
    biclusters_: np.ndarray = field(init=False)
    row_labels_: np.ndarray = field(init=False)
    column_labels_: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    best_norm: int = field(init=False)
    best_iter: int = field(init=False)

    def print_or_log(self, s):
        # NOTE: logging might break if there's an exception while logging
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def print_exit_conditions(self, local_scope):
        exit_conditions_actual = []
        exit_conditions_possible = [
            "i < self.iter_max",
            "current_norm <= previous_norm",
            [
                "abs(current_norm - previous_norm) > 0.0001*mean", 
                "abs(current_norm - previous_norm) > 0.0001*previous_norm", 
                "abs(current_norm - previous_norm) > previous_norm/higher_dim**2",
            ],
        ]

        # calculate conditions that are no longer valid
        for i, cond in enumerate(exit_conditions_possible):
            if type(cond) != str:
                for j, subcond in enumerate(cond):
                    if not eval(subcond, local_scope):
                        exit_conditions_actual.append((f"{i+1}.{j+1}", subcond))
            else:
                if not eval(cond, local_scope):
                    exit_conditions_actual.append((f"{i+1}", cond))
        
        # print valid conditions
        self.print_or_log("exit conditions (all False):")
        for index, condition in exit_conditions_actual:
            self.print_or_log(f"  {index}) {condition}")
        self.print_or_log("\n")

    #@njit
    def attempt_coclustering(self, R,B,C,Z, save_history=False):
        # NOTE: convergence check helps prevent nan-land
        i, previous_norm, current_norm = 0, np.inf, np.inf
        higher_dim, mean = max(Z.shape), Z.mean()
        if save_history:
            self.current_history = deque()
            self.current_history.append((R.copy(),B.copy(),C.copy()))
        # NOTE: last condition: if we accidentally do a oopsie and increase the norm, we exit immediately and hope everything's okay
        while (i == 0 or abs(current_norm - previous_norm) > 0.0001 or abs(current_norm - previous_norm) > 0.0001*previous_norm or abs(current_norm - previous_norm) > previous_norm/higher_dim**2) and i < self.iter_max and current_norm <= previous_norm:
            R[:,:] = R[:,:] * (Z@C.T@B.T)[:,:] / (R@B@C@C.T@B.T)[:,:]
            B[:,:] = B[:,:] * (R.T@Z@C.T)[:,:] / (R.T@R@B@C@C.T)[:,:]
            C[:,:] = C[:,:] * (B.T@R.T@Z)[:,:] / (B.T@R.T@R@B@C)[:,:]
            previous_norm = current_norm
            current_norm = np.linalg.norm(R@B@C - Z)
            if save_history:
                self.current_history.append((R.copy(),B.copy(),C.copy()))
            i += 1
        #exit_conditions = self.print_exit_conditions(local_scope=locals().copy()) ## DBG
        
        return ((R, B, C), current_norm, i)

    #@njit
    def attempt_coclustering_sym(self, S,B,Z):
        i, previous_norm, current_norm = 0, 0, 0
        while (i == 0 or abs(current_norm - previous_norm) > 1 or abs(current_norm - previous_norm) > 0.0001*previous_norm) and i < self.iter_max:
            S[:,:] = S[:,:] * (Z@S@B)[:,:] / (S@B@S.T@S@B)[:,:]
            B[:,:] = B[:,:] * (S.T@Z@S)[:,:] / (S.T@S@B@S.T@S)[:,:]
            previous_norm = current_norm
            current_norm = np.linalg.norm(S@B@S.T - Z)
            i += 1
        else:
            self.print_or_log(f"  early stop after {i} iterations")


    def do_things(self, Z, rng, verbose=False, save_history=False):
        n, m = Z.shape
        k, l = self.n_row_clusters, self.n_col_clusters
        attempt_no, best_norm, best_results, best_iter = 0, np.inf, None, 0

        while attempt_no < self.n_attempts:
            R, B, C = rng.random((n,k)), Z.mean() * np.ones((k,l)), rng.random((l,m))
            s = cool_header_thing()
            if verbose:
                self.print_or_log(f"\n{s}\nAttempt #{attempt_no+1}:\n{s}\n")
            results, current_norm, iter_stop = self.attempt_coclustering(R,B,C,Z,save_history=save_history)
            if verbose:
                if iter_stop < self.iter_max:
                    self.print_or_log(f"  early stop after {iter_stop} iterations")
                self.print_or_log(f"  Attempt #{attempt_no+1} norm: {current_norm}")

            ## silhouette for paranoia
            # NOTE: best norm =/> best silhouette but it shouldnt be too different, right?
            _, row_labels, col_labels = NBVD_coclustering.get_stuff(results[0], results[2])
            sil_row = silhouette_score(Z, row_labels)
            sil_col = silhouette_score(Z.T, col_labels)
            if verbose:
                self.print_or_log(f"  Attempt #{attempt_no+1} silhouette:\n\trows: {sil_row:.3f}\n\tcols: {sil_col:.3f}")

            if current_norm < best_norm:
                if verbose:
                    self.print_or_log("__is__ best!")
                if save_history:
                    self.best_history = self.current_history
                best_results, best_norm, best_iter = results, current_norm, iter_stop
            attempt_no += 1
        
        # set attributes so we have more info
        self.best_norm, self.best_iter = best_norm, best_iter
        return best_results

    def get_stuff (R, C):
        """Get bicluster boolean matrix, row labels and column labels from row-coefficient and
        column-coefficient matrices (R and C, respectively)."""
        n, k = R.shape
        l, m = C.shape
        row = np.argmax(R, axis=1)
        col = np.argmax(C, axis=0)

        zeros_row = np.zeros((n,k))
        _, j_idx = np.mgrid[slice(zeros_row.shape[0]), slice(zeros_row.shape[1])] # prefer anything over for loop
        row_mod = row.reshape((n,1)).repeat(k, axis=1)
        bic_rows = np.where((j_idx == row_mod) , True, False)

        zeros_col = np.zeros((l,m))
        i_idx, _ = np.mgrid[slice(zeros_col.shape[0]), slice(zeros_col.shape[1])] # prefer anything over for loop
        col_mod = col.reshape((m,1)).repeat(k, axis=1)
        bic_cols = np.where((i_idx.T == col_mod) , True, False)
        bic = (bic_rows.T, bic_cols.T)
        return (bic, row, col)


    # run after auto-generated init
    def __post_init__(self):
        # initialization
        rng = np.random.default_rng(seed=self.random_state)
        Z = np.array(self.data)

        # do stuff
        if self.symmetric:
            if k != l: 
                raise Exception("number of row clusters is different from number of column clusters")
            S = R
            NBVD_coclustering.do_things_sym(S, B, Z)
            self.S, self.B = S, B
            self.biclusters_, self.row_labels_, self.column_labels_ = NBVD_coclustering.get_stuff(S, S.T)

        else:
            self.R, self.B, self.C = self.do_things(Z, rng, verbose=self.verbose, save_history=self.save_history)
            self.biclusters_, self.row_labels_, self.column_labels_ = NBVD_coclustering.get_stuff(self.R, self.C)


        """ # warn about using symmetric instead
            if k == l:
                mat_debug(R-C.T, "pls no equal")
        """
