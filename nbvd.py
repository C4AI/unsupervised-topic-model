#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from collections import deque
import logging
from typing import Tuple
from my_utils import *


@dataclass(eq=False)
class NBVD_coclustering:
    # init arguments
    data: np.ndarray
    n_row_clusters: int
    n_col_clusters: int
    symmetric: bool = False
    iter_max: int = 2000
    n_attempts: int = 5
    random_state: int = None
    verbose: bool = False
    save_history: bool = False
    save_norm_history: bool = False
    logger : logging.Logger = None

    # properties calculated post init
    Z: np.ndarray = field(init=False)
    biclusters_: np.ndarray = field(init=False)
    row_labels_: np.ndarray = field(init=False)
    column_labels_: np.ndarray = field(init=False)
    cluster_assoc: np.ndarray = field(init=False)
    centroids: Tuple[np.ndarray] = field(init=False)
    basis_vectors: Tuple[np.ndarray] = field(init=False)
    R: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    best_norm: int = field(init=False)

    def print_or_log(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def get_basis_vectors (R,B,C):
        # R = (n,k)
        # B = (k,l)
        # C = (l,m)
        # RB = (n,l) 
        #       l column-centroids (basis vectors for Z's column space)
        # (BC).T = (m,k)
        ##      k row-centroids (basis vectors for Z's row space)

        col_basis = R @ B
        row_basis = (B @ C).T
        return (row_basis, col_basis)
    
    def get_centroids (Z, rc_labels, n_rc_clusters : Tuple[int,int]):
        row_labels, col_labels = rc_labels
        n_row_clusters, n_col_clusters = n_rc_clusters
        row_centroids = get_centroids_by_cluster(Z, row_labels, n_row_clusters)
        col_centroids = get_centroids_by_cluster(Z.T, col_labels, n_col_clusters)
        return (row_centroids, col_centroids)

    def get_labels (Z, rc_centroids, n_centroids, centroid_dim, other_centroid_dim, R=None, C=None, method="centroids"):
        if method == "centroids":
            Z_rc_extra = Z.reshape(*Z.shape, 1).repeat(n_centroids, axis=2) # add extra dim for clusters
            c_rc_extra = rc_centroids.T.reshape(n_centroids, centroid_dim, 1).repeat(other_centroid_dim, axis=2).T # add extra dim for number of samples
            rc_distances = norm(Z_rc_extra-c_rc_extra, axis=1)
        elif method == "rbc":
            if R is None or C is None:
                raise Exception("[NBVD.get_labels] Please provide R and C to use the 'rbc' method")
            row = np.argmax(R, axis=1)
            col = np.argmax(C, axis=0)
        else:
            raise Exception(f"[NBVD.get_labels] ERROR: invalid labeling method: {method}")
        return np.argmin(rc_distances, axis=1)

    def get_stuff (R, C, B=None, Z=None, centroids=None, method="centroids"):
        """Get bicluster boolean matrix, row labels and column labels from row-coefficient and
        column-coefficient matrices (R and C, respectively)."""

        n, k = R.shape
        l, m = C.shape
        if method == "rbc":
            row = np.argmax(R, axis=1)
            col = np.argmax(C, axis=0)
        elif method == "fancy": # TODO: this only works if n_row_clusters == n_col_clusters 
            U = R.copy()
            V = C.T.copy()
            diag = lambda M : np.diag(np.diag(M))
            Du = diag(np.ones(U.shape).T @ U)
            Dv = diag(np.ones(V.shape).T @ V)

            U = U @ diag(B @ Dv @ np.ones(Dv.shape))
            V = V @ diag(np.ones(Du.shape).T @ Du @ B)
            row = np.argmax(U, axis=1) # U is associated with rows; V is associated with columns
            col = np.argmax(V, axis=1)
        elif method == "centroids":
            row_centroids, col_centroids = centroids
            m, k = row_centroids.shape
            n, l = col_centroids.shape

            row = NBVD_coclustering.get_labels(Z, row_centroids, k, m, n)
            col = NBVD_coclustering.get_labels(Z.T, col_centroids, l, n, m)

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

    def get_cluster_assoc (R,B,C):
        """Get co-cluster structure from factorization results."""
        k, l = B.shape
        B_row_avg = np.average(B, axis=1) # NOTE: chosen so that each document cluster is associated w/ something
        return (B >= B_row_avg.reshape((k,1)).repeat(l, axis=1))

    def attempt_coclustering_aux(Z,R,B,C, symmetric):
        if not symmetric:
            R[:,:] = R[:,:] * (Z@C.T@B.T)[:,:] / (R@B@C@C.T@B.T)[:,:]
            B[:,:] = B[:,:] * (R.T@Z@C.T)[:,:] / (R.T@R@B@C@C.T)[:,:]
            C[:,:] = C[:,:] * (B.T@R.T@Z)[:,:] / (B.T@R.T@R@B@C)[:,:]
        else:
            S = R
            S[:,:] = S[:,:] * (Z@S@B)[:,:] / (S@B@S.T@S@B)[:,:]
            B[:,:] = B[:,:] * (S.T@Z@S)[:,:] / (S.T@S@B@S.T@S)[:,:]

    def attempt_coclustering(self, Z,R,B,C, symmetric=False):
        i, previous_norm, current_norm = 0, np.inf, np.inf

        if self.save_norm_history:
            self.current_norm_history = []
        if self.save_history:
            self.current_history = deque()
            self.current_history.append((R.copy(),B.copy(),C.copy()))

        while i == 0 or (i < self.iter_max and current_norm <= previous_norm):
            NBVD_coclustering.attempt_coclustering_aux(Z,R,B,C, symmetric=symmetric)

            previous_norm = current_norm
            current_norm = norm(R@B@C - Z)
            if self.save_norm_history:
                self.current_norm_history.append(current_norm)
            if self.save_history:
                self.current_history.append((R.copy(),B.copy(),C.copy()))
            i += 1
        
        return ((R, B, C), current_norm, i)

    def do_things(self, Z, symmetric, rng, verbose=False):
        n, m = Z.shape
        k, l = self.n_row_clusters, self.n_col_clusters
        attempt_no, best_norm, best_results, best_iter, best_sil = 0, np.inf, None, 0, MeanTuple(-np.inf)

        while attempt_no < self.n_attempts:
            if not symmetric:
                R, B, C = rng.random((n,k)), Z.mean() * np.ones((k,l)), rng.random((l,m))
            else:
                R, B = rng.random((n,k)), Z.mean() * np.ones((k,l))
                C = R.T
            s = cool_header_thing()
            if verbose:
                self.print_or_log(f"\n{s}\nAttempt #{attempt_no+1}:\n{s}\n")
            
            results, current_norm, iter_stop = self.attempt_coclustering(Z,R,B,C, symmetric=symmetric)

            if verbose:
                if iter_stop < self.iter_max:
                    self.print_or_log(f"  early stop after {iter_stop} iterations")
                self.print_or_log(f"  Attempt #{attempt_no+1} norm: {current_norm}")

            R,B,C = results
            basis = NBVD_coclustering.get_basis_vectors(R, B, C)
            _, row_labels, col_labels = NBVD_coclustering.get_stuff(R, C, B=B, Z=Z, method="fancy")
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
        
    # runs after auto-generated init
    def __post_init__(self):
        # initialization
        rng = default_rng(seed=self.random_state)
        self.data = np.array(self.data)
        Z = self.data
        self.Z = self.data # in case we prefer to call it this way

        # clustering # /DEL
        if self.symmetric: # /DEL
            n,m = Z.shape
            if n != m: 
                raise Exception("Number of row clusters is different from number of column clusters.")
        self.R, self.B, self.C = self.do_things(Z, symmetric=self.symmetric, rng=rng, verbose=self.verbose)
        if self.symmetric: # /DEL
            self.S = self.R

        self.basis_vectors = NBVD_coclustering.get_basis_vectors(self.R, self.B, self.C)
        self.biclusters_, self.row_labels_, self.column_labels_ = NBVD_coclustering.get_stuff(
                            self.R, self.C, self.B, self.data, 
                            method="fancy")
        self.cluster_assoc = NBVD_coclustering.get_cluster_assoc(self.R, self.B, self.C)
        self.centroids = NBVD_coclustering.get_centroids(
                        self.data, 
                        (self.row_labels_, self.column_labels_), 
                        (self.n_row_clusters, self.n_col_clusters)
                        )

