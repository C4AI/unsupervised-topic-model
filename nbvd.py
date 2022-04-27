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
from typing import Tuple

from my_utils import *

# TODO: fix first exit condition
# TODO: decide which labeling thing to use

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
    save_norm_history: bool = False
    logger : logging.Logger = None

    # properties calculated post init
    biclusters_: np.ndarray = field(init=False)
    row_labels_: np.ndarray = field(init=False)
    column_labels_: np.ndarray = field(init=False)
    cluster_assoc: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    best_norm: int = field(init=False)
    best_norm: int = field(init=False)
    centroids: Tuple[np.ndarray] = field(init=False)

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
    def attempt_coclustering(self, R,B,C,Z):
        # NOTE: convergence check helps prevent nan-land
        i, previous_norm, current_norm = 0, np.inf, np.inf
        higher_dim, mean = max(Z.shape), Z.mean()
        self.current_norm_history = []
        if self.save_history:
            self.current_history = deque()
            self.current_history.append((R.copy(),B.copy(),C.copy()))

        #while (i == 0 or abs(current_norm - previous_norm) > 0.00001 or abs(current_norm - previous_norm) > 0.0001*previous_norm or abs(current_norm - previous_norm) > previous_norm/higher_dim**2) and i < self.iter_max and current_norm <= previous_norm:
        while i == 0 or (i < self.iter_max and current_norm <= previous_norm):
            R[:,:] = R[:,:] * (Z@C.T@B.T)[:,:] / (R@B@C@C.T@B.T)[:,:]
            B[:,:] = B[:,:] * (R.T@Z@C.T)[:,:] / (R.T@R@B@C@C.T)[:,:]
            C[:,:] = C[:,:] * (B.T@R.T@Z)[:,:] / (B.T@R.T@R@B@C)[:,:]
            previous_norm = current_norm
            current_norm = np.linalg.norm(R@B@C - Z)
            self.current_norm_history.append(current_norm)
            #print(f"{current_norm} {previous_norm}")
            if self.save_history:
                self.current_history.append((R.copy(),B.copy(),C.copy()))
            i += 1
        #exit_conditions = self.print_exit_conditions(local_scope=locals().copy()) ## DBG
        
        return ((R, B, C), current_norm, i)

    #@njit
    def attempt_coclustering_sym(self, S,B,Z):
        # NOTE: convergence check helps prevent nan-land
        i, previous_norm, current_norm = 0, np.inf, np.inf
        higher_dim, mean = max(Z.shape), Z.mean()
        if self.save_history:
            self.current_history = deque()
            self.current_history.append((S.copy(),B.copy(),S.T.copy()))
        # NOTE: last condition: if we accidentally do a oopsie and increase the norm, we exit immediately and hope everything's okay
        while (i == 0 or abs(current_norm - previous_norm) > 0.00001 or abs(current_norm - previous_norm) > 0.0001*previous_norm or abs(current_norm - previous_norm) > previous_norm/higher_dim**2) and i < self.iter_max and current_norm <= previous_norm:
            S[:,:] = S[:,:] * (Z@S@B)[:,:] / (S@B@S.T@S@B)[:,:]
            B[:,:] = B[:,:] * (S.T@Z@S)[:,:] / (S.T@S@B@S.T@S)[:,:]
            previous_norm = current_norm
            current_norm = np.linalg.norm(S@B@S.T - Z)
            #print(f"{current_norm} {previous_norm}")
            if self.save_history:
                self.current_history.append((S.copy(),B.copy(),S.T.copy()))
            i += 1
        #exit_conditions = self.print_exit_conditions(local_scope=locals().copy()) ## DBG
        
        return ((S, B, S.T), current_norm, i)


    def do_things(self, Z, symmetric, rng, verbose=False):
        n, m = Z.shape
        k, l = self.n_row_clusters, self.n_col_clusters
        attempt_no, best_norm, best_results, best_iter, best_sil = 0, np.inf, None, 0, MeanTuple(-np.inf)

        while attempt_no < self.n_attempts:
            if not symmetric:
                R, B, C = rng.random((n,k)), Z.mean() * np.ones((k,l)), rng.random((l,m))
            else:
                S, B = rng.random((n,k)), Z.mean() * np.ones((k,l))
            s = cool_header_thing()
            if verbose:
                self.print_or_log(f"\n{s}\nAttempt #{attempt_no+1}:\n{s}\n")
            
            if not symmetric:
                results, current_norm, iter_stop = self.attempt_coclustering(R,B,C,Z)
            else:
                results, current_norm, iter_stop = self.attempt_coclustering_sym(S,B,Z)
            if verbose:
                if iter_stop < self.iter_max:
                    self.print_or_log(f"  early stop after {iter_stop} iterations")
                self.print_or_log(f"  Attempt #{attempt_no+1} norm: {current_norm}")

            ## silhouette for paranoia
            # NOTE: best norm =/> best silhouette but it shouldnt be too different, right?
            _, row_labels, col_labels = NBVD_coclustering.get_stuff(results[0], results[2], B=results[1], method="not fancy")
            sil_row = silhouette_score(Z, row_labels)
            sil_col = silhouette_score(Z.T, col_labels)
            silhouette = MeanTuple(sil_row, sil_col)
            if verbose:
                self.print_or_log(f"  Attempt #{attempt_no+1} silhouette:\n\trows: {sil_row:.3f}\n\tcols: {sil_col:.3f}")

            if silhouette > best_sil:
                if verbose:
                    self.print_or_log("__is__ best!")
                if self.save_history:
                    self.best_history = self.current_history
                if self.save_norm_history:
                    self.norm_history = self.current_norm_history
                best_results, best_norm, best_iter, best_sil = results, current_norm, iter_stop, silhouette
            attempt_no += 1
        
        # set attributes so we have more info
        self.best_norm, self.best_iter = best_norm, best_iter
        return best_results

    def get_centroids (self):
        ## R = (n,k)
        ## B = (k,l)
        ## C = (l,m)
        ## RB = (n,l) l centroides de colunas (vetores-base p espaco de colunas de Z)
        ## (BC).T = (m,k) k centroides de linhas (vetores-base p espaco de linhas de Z)
        col_centroids = self.R @ self.B
        row_centroids = (self.B @ self.C).T
        return (row_centroids, col_centroids)

    def get_stuff (R, C, B=None, Z=None, centroids=None, method="centroids"):
        """Get bicluster boolean matrix, row labels and column labels from row-coefficient and
        column-coefficient matrices (R and C, respectively)."""

        n, k = R.shape
        l, m = C.shape
        if method == "not fancy":
            row = np.argmax(R, axis=1)
            col = np.argmax(C, axis=0)
        elif method == "fancy":
            U = R.copy()
            V = C.T.copy()
            diag = lambda M : np.diag(np.diag(M))
            Du = diag(np.ones(U.shape).T @ U)
            Dv = diag(np.ones(V.shape).T @ V)

            U = U @ diag(B @ Dv @ np.ones(Dv.shape))
            V = V @ diag(np.ones(Du.shape).T @ Du @ B)
            row = np.argmax(U, axis=1) # NOTE: U is associated with rows; V is associated with columns
            col = np.argmax(V, axis=1)
        elif method == "centroids":
            # TODO: clean up
            row_centroids, col_centroids = centroids
            m, k = row_centroids.shape
            n, l = col_centroids.shape
            

            ###row_distances = norm(Z.T-row_centroids)
            """ works i think
            Amod = A.reshape(5,3,1).repeat(3, axis=2)
            cmod=c.reshape(3,3,1).T.repeat(5,axis=0) # add extra dim ,transpose and repeat
            row_distances = norm(Zmod-cmod,axis=1)
            """
            """ old col
            Z_col_extra = Z.T.reshape(*Z.T.shape, 1).repeat(l, axis=2) # add extra dim for clusters
            #c_col_extra = col_centroids.reshape(n, l, 1).T.repeat(n, axis=0) # add extra dim for number of samples
            c_col_extra = col_centroids.reshape(n, 1, l).repeat(n, axis=1) # add extra dim for number of samples
            col_distances = norm(Z_col_extra-c_col_extra, axis=1)
            col = np.argmin(col_distances, axis=1)
            """

            Z_row_extra = Z.reshape(*Z.shape, 1).repeat(k, axis=2) # add extra dim for clusters
            #c_row_extra = row_centroids.reshape(m, k, 1).T.repeat(m, axis=0) # add extra dim for number of samples
            #c_row_extra = row_centroids.reshape(m, 1, k).repeat(m, axis=1) # add extra dim for number of samples
            
            # NOTE: i swapped '.repeat(m' for '.repeat(n
            c_row_extra = row_centroids.T.reshape(k, m, 1).repeat(n, axis=2).T # add extra dim for number of samples
            row_distances = norm(Z_row_extra-c_row_extra, axis=1)
            row = np.argmin(row_distances, axis=1)
            
            Z_col_extra = Z.T.reshape(*Z.T.shape, 1).repeat(l, axis=2) # add extra dim for clusters
            #c_col_extra = col_centroids.reshape(n, l, 1).T.repeat(n, axis=0) # add extra dim for number of samples
            # NOTE: i swapped '.repeat(n' for '.repeat(m
            c_col_extra = col_centroids.T.reshape(l, n, 1).repeat(m, axis=2).T # add extra dim for number of samples
            col_distances = norm(Z_col_extra-c_col_extra, axis=1)
            col = np.argmin(col_distances, axis=1)

            """
            # DBG EXCLUDE sanity check
            row2, col2 = np.zeros((n,), dtype="int64"), np.zeros((m,), dtype="int64")
            for i, r in enumerate(Z):
                distances = [norm(r-centroid) for centroid in row_centroids.T]
                row2[i] = np.argmin(distances)
            for i, c in enumerate(Z.T):
                distances = [norm(c-centroid) for centroid in col_centroids.T]
                col2[i] = np.argmin(distances)
            print("row centroids looking ok?? ", np.sum(row == row2))
            print("col centroids looking ok?? ", np.sum(col == col2))
            """

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

    def get_cluster_assoc (self):
        R, B, C = self.R, self.B, self.C
        k, l = B.shape
        B_row_avg = np.average(B, axis=1) # NOTE: chosen so that each document cluster is associated w/ something
        self.cluster_assoc = (B >= B_row_avg.reshape((k,1)).repeat(l, axis=1))
        

    # run after auto-generated init
    def __post_init__(self):
        # initialization
        rng = np.random.default_rng(seed=self.random_state)
        self.data = np.array(self.data)
        Z = self.data

        # do stuff
        if self.symmetric:
            k,l = Z.shape
            if k != l: 
                raise Exception("number of row clusters is different from number of column clusters")
            self.S, self.B, _ = self.do_things(Z, symmetric=True, rng=rng, verbose=self.verbose)
            self.biclusters_, self.row_labels_, self.column_labels_ = NBVD_coclustering.get_stuff(self.S, self.S.T)
            self.R, self.C = self.S, self.S.T
        else:
            self.R, self.B, self.C = self.do_things(Z, symmetric=False, rng=rng, verbose=self.verbose)
            self.centroids = self.get_centroids()
            self.biclusters_, self.row_labels_, self.column_labels_ = NBVD_coclustering.get_stuff(self.R, self.C, self.B, self.data, self.centroids)
            self.get_cluster_assoc() # for cocluster analysis

        # TODO: warn about using symmetric instead
        """ 
            if k == l:
                mat_debug(R-C.T, "pls no equal")
        """
