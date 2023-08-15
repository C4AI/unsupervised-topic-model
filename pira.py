#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
import math
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.datasets import make_biclusters
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, consensus_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import nltk
import colored
from typing import Literal, Iterable # Python 3.8+
from collections import Counter, OrderedDict # OrderedDict is redundant as of Python 3.7
import sys,os,pickle,re
from itertools import combinations

from nbvd import NBVD_coclustering
from my_utils import *

try:
    stop_words_nltk=nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words_nltk=nltk.corpus.stopwords.words('english')

stop_words_nltk.extend(['also','semi','multi','sub','non','et','al','like','pre','post', # preffixes
    'ltd','sa','SA','copyright','eg','etc','elsevier','springer','springer_verlag','inc','publishing','reserved', # copyright
    'g','m','kg','mg','mv','km','km2','cm','bpd','bbl','cu','ca','mday','yr','per', # units
    'th','one','two','three','four','five','six','seven','eight','nine','ten','de','within','previously','across','top','may','mainly','thus','highly','due','including','along','since','many','various','however','could', # misc 1
    'end','less','able','according','include','included','around','last','first','major','set','average','total','new','based','different','main','associated','related','regarding','approximately','others', # misc 2
    'likely','later','would','together','even','part','using','mostly','several','values','important','although', # misc 3
    'study','studies','studied','research','paper','suggests','suggest','indicate','indicates','show','shows','result','results','present','presents','presented','consider','considered','considering','proposed','discussed','licensee','authors','aims', # research jargon 1
    'analysis','obtained','estimated','observed','data','model','sources','revealed','found','problem','used','article', # research jargon 2

])
  
# TODO: double-check nothing breaks and:
# add: 'os','nature','algorithm','poorly','strongly','universidade','years','yr','showed', 
# possibly some meaning: bpd,bbl (barrel per day -> petroleum), Cu (copper), Ca (calcium), rights (to drilling)
# TODO: regex for copyright

N_ROW_CLUSTERS, N_COL_CLUSTERS = 4,4
RNG_SEED=423
VECTORIZATION='tfidf'
vec_kwargs = Bunch(min_df=4, stop_words=stop_words_nltk, lowercase=False)

ITER_MAX=2000
N_ATTEMPTS=1 # DBG
W2V_DIM=100
ALG='nbvd'
WAIT_TIME = 4 # wait time between tasks
rerun_embedding=True
LABELING_METHOD="centroids method"
CLUSTER_CENTER_IS_AVERAGE=False
DEFAULT_CLUSTER_CENTER_METHOD = "cluster_avgs" if CLUSTER_CENTER_IS_AVERAGE else "prototype_centers"
KEEP_WORD_REPS=True # required for centroid method (were reusing the column centroid after all)

LABEL_CHECK = True
SHADE_CENTROIDS = True
SHADE_COCLUSTERS = True
NORM_PLOT = False # (NBVD) display norm plot
MOVIE=False
ASPECT_RATIO=4 # 1/6 for w2v; 10 for full tfidf; 4 for partial
SHOW_IMAGES=True
NEW_ABS=True
LOG_BASE_FOLDER = "classification_info"

REP_METHOD_FOR_ORIG_ABS="matrix_assoc_fancy"

############################################################################## 
# to use a set number of cpus: 
#   taskset --cpu-list 0-7 python "pira.py"
##############################################################################

class FooClass:
    pass
exp_numbers = re.compile("[^A-Za-z]\d+\.\d+|[^A-Za-z]\d+\,\d+|[^A-Za-z]\d+")
exp_non_alpha = re.compile("[^A-Za-zÀàÁáÂâÃãÉéÊêÍíÓóÔôÕõÚúÇç02-9 \._–+]+")
exp_whitespace = re.compile("\s+")
exp_hyphen = re.compile("(?<=[a-z])\-(?=[a-z])")

# NOTES: no overly small abstracts (all greater than 300 characters); 
# but there are some duplicates
class Preprocessor:
    def lower_but_keep_acronyms (s):
        new = []
        for w in s.split(" "):
            new.append(w if w.isupper() and len(w) >= 2 else w.lower())
        return " ".join(new)

    def preprocess (self, sentence):
        new_sentence = sentence
        new_sentence = Preprocessor.lower_but_keep_acronyms(new_sentence)
        new_sentence = re.sub(exp_hyphen, "_", new_sentence) # keep compound words in tokenization
        new_sentence = re.sub(exp_numbers, " 1", new_sentence)
        new_sentence = re.sub(exp_non_alpha, "", new_sentence)
        new_sentence = re.sub(exp_whitespace, " ", new_sentence)
        return new_sentence

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_list = X.to_list() if type(X) != list else X
        unique_sentences = set()
        newX = []
        for i,sentence in enumerate(X_list):
            if len(sentence) > 300 and sentence not in unique_sentences:
                unique_sentences.add(sentence)
                newX.append(self.preprocess(sentence))
        return newX

def __candidate_selection (dists_to_centroid, labels, cluster_no, n_representatives):
    count = 0
    for i, _ in dists_to_centroid:
        if labels[i] == cluster_no:
            count += 1
            yield i
        if count >= n_representatives:
            return # stop yielding

def get_representatives (data, model, n_clusters, n_representatives=5, reverse=False, 
        method='centroid_dif', metric='cosine', kind=None, 
        cluster_center_method=DEFAULT_CLUSTER_CENTER_METHOD) -> dict:
    cluster_representatives = {}

    # get relevant properties
    if kind == 'docs':
        labels = model.row_labels_
    elif kind == 'words':
        labels = model.column_labels_
    else:
        raise Exception("get_representatives: must specify 'kind'")
    original_data = model.__original_data if hasattr(model, "__original_data") else None
    vec = model.__vectorization if hasattr(model, "__vectorization") else None

    if hasattr(model, "B"):
        R,B,C = model.R, model.B, model.C

    print(f"[get_representatives] using {method} as method") # DBG

    if method == 'centroid_dif':
        print(f"[get_representatives] using {metric} metric") # DBG
        # choose cluster centers
        if kind == 'docs':
            elements_index = 0
        elif kind == 'words':
            elements_index = 1
        if cluster_center_method == "prototype_centers" and method == "centroid_dif":
            print("[get_representatives] using prototype centers as representatives")
            normalized_centroids = normalize(model.basis_vectors[elements_index].T)
        elif cluster_center_method == "cluster_avgs" and method == "centroid_dif":
            print("[get_representatives] using (old) cluster avgs as representatives")
            normalized_centroids = normalize(model.centroids[elements_index].T)
            #centroids_ = get_centroids_by_cluster(data, labels, n_clusters) # OLD # DBG
        else:
            raise Exception(f"[get_representatives] invalid center method: {cluster_center_method}")

        # calculate distances to centroids
        # NOTE: words are not normalized (but documents are)
        all_distances = np.zeros((data.shape[0], n_clusters))
        n_data = normalize(data) if kind == 'words' else data # TODO: comment out if
        # TODO: test linear_kernel
        all_distances = pairwise_distances(n_data, normalized_centroids, metric=metric)
        print(f"all_distances shape: {all_distances.shape =}") # DBG
    elif method == 'matrix_assoc' or method == 'matrix_assoc_fancy':
        reverse = not reverse # small distance == big assoc
        adh_method = "fancy" if ("fancy" in method) else "rbc"
        row_adh, col_adh = NBVD_coclustering.get_adherence(R, C, B, method=adh_method)
        if kind == 'docs':
            all_distances = row_adh
        elif kind == 'words':
            all_distances = col_adh
    else:
        raise Exception(f"[get_representatives] invalid method: {method}")

    # get representatives
    for c in range(n_clusters):
        dists_to_centroid = sorted(
            zip(range(data.shape[0]), list(all_distances[:,c])), 
            key = lambda t : t[1], reverse=reverse
        )
        # select top n; eliminate candidates that arent from the relevant cluster
        rep_candidates = list(__candidate_selection(dists_to_centroid, labels, c, n_representatives))
        if rep_candidates: # if anyones left
            cluster_representatives[c] = rep_candidates   
    
    return cluster_representatives

def calculate_occurrence (word, original_data, indices):
    count = 0
    for i in indices:
        if word.lower() in original_data[i].lower():
            count += 1
    return count

def cluster_summary (data, model, n_doc_reps=5, n_word_reps=20, n_frequent=40,
            word_reps=None, verbose=True, logger=None, rep_method="centroid_dif"):
    """Most representative documents/words are chosen based on distance to the (squashed) average of the assigned cluster.
    w_occurrence_per_d_cluster is a dict with length equal to n_word_reps*n_col_clusters and values corresponding to a dict of Bunch
    of the form 
        (occ = (occurence in top documents for d_cluster, occurence in bottom documents for d_cluster),
        assigned_dc = associated document cluster, 
        assigned_wc = associated word cluster, .
    (If d_cluster != cocluster-associated document cluster, no occurrence in bottom documents.)
    Note: if word_reps is given, word representatives are not calculated.

    Returns: ((most representative documents, most representative words), word occurrence per document cluster)"""

    print_or_log = logger.info if logger else (print if verbose else lambda s : None)
    has_cocluster_info = hasattr(model, "cluster_assoc")
    row_centroids, col_centroids = model.centroids
    row_labels_, column_labels_ = model.row_labels_, model.column_labels_
    vec, original_data = model.__vectorization, model.__original_data
    m, k = row_centroids.shape
    n, l = col_centroids.shape
    smallest_rcluster_size = min(np.bincount(row_labels_))
    if smallest_rcluster_size <= 10:
        if logger:
            logger.warn("A document cluster has size 0..")
        else: # TODO: better warning/logging
            print("[cluster_summary] WARNING: A document cluster has size 0..")
    elif smallest_rcluster_size == 0:
        raise Exception("[cluster_summary] A cluster has size 0")
    if has_cocluster_info:
        cluster_assoc = model.cluster_assoc
        assoc_shape = cluster_assoc.shape
    else:
        print_or_log("No cocluster info.")

    if has_cocluster_info:
        # get relevant coclusters
        relevant_coclusters = []
        for i in range(assoc_shape[0]):
            for j in range(assoc_shape[1]):
                if cluster_assoc[i,j]:
                    relevant_coclusters.append((i,j))
    
    # get row- and column-cluster representatives
    row_cluster_representatives = get_representatives(data, model, k, 
                                n_representatives=n_doc_reps, kind='docs',
                                method=rep_method
                                )
    if word_reps is None: # calculate word representatives if they are not given
        col_cluster_representatives = get_representatives(data.T, model, l, 
                                    n_representatives=n_word_reps, kind='words',
                                    method=rep_method
                                    )
    else:
        col_cluster_representatives = dict([(k, reps[:n_word_reps]) for k,reps in word_reps.items()])
    model.__row_reps, model.__col_reps = row_cluster_representatives, col_cluster_representatives

    # documents
    print_or_log("DOCUMENTS:\n")
    for i, reps in sorted(row_cluster_representatives.items()):
        print_or_log("cluster:", i)
        for rep in reps:
            print_or_log(f"rep: {rep}")
            print_or_log(original_data[rep][:200])
        print_or_log("--------------------------------------------------------\n")

    if isinstance(vec, TfidfVectorizer):
        # word analysis
        print_or_log("WORDS:\n")
        idx_to_word = vec.get_feature_names()
        for i, c in sorted(col_cluster_representatives.items()):
            print_or_log("cluster:", i)
            to_print = []
            for rep in c:
                to_print.append(idx_to_word[rep])
            print_or_log(",".join(to_print))
            print_or_log("--------------------------------------------------------\n")
        
        if has_cocluster_info:
            # cocluster analysis
            print_or_log("COCLUSTERS:")
            N = n_frequent
            if smallest_rcluster_size < N:
                N = smallest_rcluster_size
            # TODO: do top 10% / 20% / 25% instead?
            # TODO: account for clusters smaller than N; duct tape solution is to reduce N manually
            print_or_log(f"word (occurrence in top {N} documents)(occurrence in bottom {N} documents) (occurrence in other doc clusters)")
            row_reps_topN = get_representatives(data, model, 
                k, n_representatives=N, kind='docs', method=rep_method)
            row_reps_bottomN = get_representatives(data, model, 
                k, n_representatives=N, reverse=True, kind='docs', method=rep_method)
            
            # for each cocluster
            w_occurrence_per_d_cluster = OrderedDict() # store occurrence and dc info for each word
            for dc, wc in relevant_coclusters:
                
                print_or_log("cocluster:", (dc, wc),"\n")
                to_print = []
                reps = col_cluster_representatives[wc] # get the representatives for the word cluster

                # for each word, calculate its occurrence in each document cluster
                for w in reps:
                    if dc not in row_reps_topN:
                        if logger:
                            logger.warn(f"## @#@ #@ {dc} not in row_reps!!!\n")
                        else:
                            print("f [WARNING] ## @#@ #@ {dc} not in row_reps!!!\n")
                        continue
                    else:
                        # occurrence for the dc in the cocluster
                        word = idx_to_word[w]
                        size_topN = len(row_reps_topN[dc])
                        size_botN = len(row_reps_bottomN[dc])
                        # NOTE: N == smallest_rcluster_size, so row_reps sizes are guaranteed be smaller
                        if size_topN < N or size_botN < N: # DBG
                            print(f"###########\n[cluster_summary] Warning: doc cluster {dc} only has {size_topN} reps; expected {N}\n###########")
                        
                        oc_top = 100/size_topN * calculate_occurrence(word, original_data, row_reps_topN[dc])
                        oc_bottom = 100/size_botN * calculate_occurrence(word, original_data, row_reps_bottomN[dc])
                        
                        w_occurrence_per_d_cluster[word] = OrderedDict()
                        w_occurrence_per_d_cluster[word][dc] = Bunch(occ=(oc_top, oc_bottom), assigned_dc=dc, assigned_wc=wc)

                        # occurrence for other dcs
                        oc_others = []
                        for rclust in sorted(row_reps_topN.keys()):
                            if rclust == dc:
                                continue
                            size_top_other = len(row_reps_topN[rclust])
                            if size_top_other < N: # DBG
                                print(f"###########\n[cluster_summary] Warning: doc cluster {rclust} only has {size_top_other} reps; expected {N}\n###########")

                            oc_other = 100/size_top_other * calculate_occurrence(word, original_data, row_reps_topN[rclust])
                            oc_others.append((rclust, oc_other))
                            w_occurrence_per_d_cluster[word][rclust] = Bunch(occ=(oc_other, ), assigned_dc=dc, assigned_wc=wc)

                        # print (later) word occurrence in each cluster
                        oc_other_str = "".join([f"({rclust}:{oc_other:.0f}%)" for rclust,oc_other in oc_others])
                        to_print.append(f"{word}(T:{oc_top:.0f}%)(B:{oc_bottom:.0f}%) {oc_other_str}")
                print_or_log(", ".join(to_print)+"\n--------------------------------------------------------\n")

    return (row_cluster_representatives, col_cluster_representatives), w_occurrence_per_d_cluster

def cocluster_words_bar_plot (w_occurrence_per_d_cluster, filename_suffix=""):
    total_words = len(w_occurrence_per_d_cluster.keys())
    any_word = list(w_occurrence_per_d_cluster.keys())[0]
    n_clusters = len(w_occurrence_per_d_cluster[any_word])
    n_word_reps = int(total_words / n_clusters)

    """ 
    if n_word_reps != n_word_reps_display_only: # DBG
        print(f"{total_words =} | {any_word =} | {n_clusters =} | {n_word_reps =}")
        raise Exception(f"{n_word_reps =} but it should be {n_word_reps_display_only}")
    """

    n_hplots, n_vplots = math.ceil(math.sqrt(n_word_reps)), round(math.sqrt(n_word_reps)) # more rows than columns
    if n_hplots * n_vplots < n_word_reps:
        raise Exception("cocluster_words_bar_plot: math?")
    
    custom_figsize = (2*6.4,2*4.8)
    #n_hplots = 4 # DBG
    #n_vplots = 6 # DBG
    # custom_figsize = (17,7.437) # DBG

    # translate a dictionary of word occurrences per cluster into bar plots
    # NOTE: w_occurrence_per_d_cluster is ordered; words are already grouped by cluster
    current_dc, current_ax = None, 1
    current_wc, previous_wc = None, None
    fig = plt.figure(figsize=custom_figsize) 
    fig.set_tight_layout(True)
    for word, info in w_occurrence_per_d_cluster.items():
        w_assigned_dc = info[0].assigned_dc # assigned_dc for w, inside info for cluster 0
        if current_dc is None: 
            current_dc = w_assigned_dc # for initial item
            current_wc = info[0].assigned_wc
            fig.suptitle(f"Word cluster {current_wc} (top {n_word_reps}): occurrence in doc clusters\n")

        # make a new figure for a different cluster
        if w_assigned_dc != current_dc:
            # save previous figure
            previous_wc = current_wc
            current_wc = info[0].assigned_wc
            fig.savefig(f"bar_plot_wc{previous_wc}_{filename_suffix}.png")
            
            current_dc = w_assigned_dc
            current_ax = 1
            fig = plt.figure(figsize=custom_figsize)
            fig.suptitle(f"Word cluster {info[0].assigned_wc} (top {n_word_reps}): occurrence in doc clusters\n")
            fig.set_tight_layout(True)
        
        # bar plot for current word
        ax = fig.add_subplot(n_hplots, n_vplots, current_ax) # subplot index is 1-based
        current_ax += 1
        short_info = sorted([(k, v.occ[0]) for k,v in info.items()]) # value = occurrence in top docs
        labels, values = zip(*short_info) # split into keys, values
        color = ["#64001E" if (l != w_assigned_dc) else "#00FA8C" for l in labels]
        ax.bar(labels, values, color=color) # categorical plot
        ax.set_title(word)
    fig.savefig(f"bar_plot_wc{current_wc}_{filename_suffix}.png") # save last figure
    plt.show()

def do_vectorization (new_abstracts, vectorization_type, **kwargs):
    if vectorization_type == 'tfidf':
        vec = TfidfVectorizer(**kwargs)
        data = vec.fit_transform(new_abstracts).toarray()
    elif vectorization_type == 'count':
        vec = CountVectorizer(**kwargs)
        data = vec.fit_transform(new_abstracts).toarray()
    elif vectorization_type == 'tfidf-char':
        #vec = TfidfVectorizer(ngram_range=(5,5), analyzer='char', max_features=15000)
        vec = TfidfVectorizer(ngram_range=(5,5), analyzer='char')
        data = vec.fit_transform(new_abstracts).toarray()
    return (data, vec)

def do_task_single (data, original_data, vectorization, only_one=True, alg=ALG, 
        show_images=True, first_image_save_path=None, RNG_SEED=None, logger=None, iter_max=ITER_MAX):
    RNG = np.random.default_rng(RNG_SEED)

    if logger:
        logger.info(f"shape: {data.shape}")
    else:
        print(f"shape: {data.shape}")
    timer = None if only_one else WAIT_TIME * show_images 

    # do co-clustering
    if alg == 'nbvd':
        model = NBVD_coclustering(data, n_row_clusters=N_ROW_CLUSTERS, 
            n_col_clusters=N_COL_CLUSTERS, n_attempts=N_ATTEMPTS, iter_max=iter_max, random_state=RNG_SEED, 
            verbose=True, save_history=MOVIE, save_norm_history=NORM_PLOT, logger=logger)

    # add extra info to model
    model.__original_data, model.__vectorization = original_data, vectorization

    print(f"B =\n{model.B}") # DBG 

    # show animation of clustering process
    if MOVIE and alg == 'nbvd':
        pyqtgraph_thing(data, model, 25)

    #########################
    # evaluate results 
    #########################

    ### internal indices
    # print silhouette scores
    silhouette = print_silhouette_score(data, model.row_labels_, model.column_labels_, logger=logger)

    if show_images:
        # shade lines/columns of original dataset
        if LABEL_CHECK:
            shaded_label_matrix(data, model.row_labels_, kind="rows", method_name=LABELING_METHOD, RNG=RNG, opacity=1, aspect_ratio=ASPECT_RATIO)
            shaded_label_matrix(data, model.column_labels_, kind="columns", method_name=LABELING_METHOD, RNG=RNG, opacity=1, aspect_ratio=ASPECT_RATIO)
            if SHADE_COCLUSTERS and alg == "nbvd":
                shade_coclusters(data, (model.row_labels_, model.column_labels_), 
                    model.cluster_assoc, RNG=RNG, aspect_ratio=ASPECT_RATIO)
        
        # centroid (and dataset) (normalized) scatter plot
        if SHADE_CENTROIDS and alg == 'nbvd':
            row_centroids, col_centroids = model.centroids[0], model.centroids[1]
            model.row_pca, model.row_c_palette, _ = centroid_scatter_plot(
                    data, 
                    row_centroids, 
                    model.row_labels_,
                    title="Rows and Row centroids", 
                    basis_vectors=model.basis_vectors[0],
                    save_path="doc_scatter_plot.png",
                    RNG=RNG
                )
            model.col_pca, model.col_c_palette, _ = centroid_scatter_plot(
                    data.T, 
                    col_centroids, 
                    model.column_labels_, 
                    title="Columns and Column centroids", 
                    basis_vectors=model.basis_vectors[1], 
                    save_path="word_scatter_plot.png",
                    RNG=RNG
            )
        
        # norm evolution
        if hasattr(model, "norm_history"):
            plot_norm_history(model)

        # general plots #/DEL
        if alg == 'nbvd':
            to_plot = [data, model.R@model.B@model.C]
            names = ["Original dataset", "Reconstructed matrix RBC"]
        # TODO: make this nicer maybe keep the timer logic
        #plot_matrices(to_plot, names, timer = None if only_one else 2*timer, aspect_ratio=ASPECT_RATIO)
        plt.show()

    # textual analysis
    representatives, w_occurrence_per_d_cluster = cluster_summary(data, model, 
                                                rep_method=REP_METHOD_FOR_ORIG_ABS)
    
    if show_images:
        cocluster_words_bar_plot(w_occurrence_per_d_cluster, filename_suffix="orig_abs")

    # return general statistics
    if alg == 'nbvd':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            best_iter=model.best_iter, best_norm=model.best_norm, n_attempts=N_ATTEMPTS)
    else:
        bunch = Bunch(silhouette=MeanTuple(*silhouette), n_attempts=N_ATTEMPTS)
    return (model, bunch)

def load_new_new_abstracts (path, n_abstracts, old_abstracts):
    old_abstracts_S = set(old_abstracts)
    df = pd.read_csv(path, delimiter=',')
    new_new_abstracts = df['abstract'][:n_abstracts].to_list()
    new_new_not_repeat = [ab for ab in new_new_abstracts if ab not in old_abstracts_S]
    new_processed_abstracts = Preprocessor().transform(new_new_not_repeat) # preprocess and eliminate duplicates
    print(f"\nnew abstracts: {len(new_processed_abstracts)} | old abstracts present: {len(new_new_abstracts) - len(new_new_not_repeat)}")
    return new_processed_abstracts, df

def vec_and_class_new_abstracts (extra_abstracts : Iterable, vec, model, logger=None, verbose=False):
    print_or_log = logger.info if logger else print
    if CLUSTER_CENTER_IS_AVERAGE:
        row_centers, col_centers = model.centroids
    else:
        row_centers, col_centers = model.basis_vectors
    m, k = row_centers.shape
    n, l = col_centers.shape

    # vectorize abstracts
    Z = vec.transform(extra_abstracts).toarray()
    n, _ = Z.shape

    # classify rows and columns
    row_classification = NBVD_coclustering.get_labels_new_data(Z, row_centers, k, m, n)
    col_classification = model.column_labels_
    return (Z, row_classification, col_classification)

def new_abs_reduced_centroids_plot (Z, new_labels, orig_model, RNG=None):
    RNG = RNG or np.default_rng()
    # plot new samples and old cluster averages
    old_row_centroids = get_centroids_by_cluster(orig_model.data, orig_model.row_labels_, n_clusters=orig_model.centroids[0].shape[1])
    _, _, ax = centroid_scatter_plot(
        Z, old_row_centroids, new_labels, 
        title="New samples and Row centroids", pca=orig_model.row_pca,
        palette=orig_model.row_c_palette, save_path="doc_new_scatter_plot.png", RNG=RNG,
        basis_vectors = orig_model.basis_vectors[0]
    )

    # plot new cluster averages
    new_centroids = get_centroids_by_cluster(Z, new_labels, n_clusters=orig_model.centroids[0].shape[1])
    new_points = normalize(new_centroids.T, axis=1)
    reduced_new_points = orig_model.row_pca.transform(new_points)
    for i, r_centroid in enumerate(reduced_new_points):
        ax.scatter(*r_centroid, color=orig_model.row_c_palette[i], marker="*", s=700, alpha=0.8)            
    
    # plot prototype centers
    m_prot_centers = orig_model.basis_vectors[0]
    new_points = normalize(m_prot_centers.T, axis=1)
    reduced_new_points = orig_model.row_pca.transform(new_points)
    for i, r_centroid in enumerate(reduced_new_points):
        ax.scatter(*r_centroid, color=orig_model.row_c_palette[i], marker="o", s=700, alpha=0.8)
    
    # legend for the centroids
    handles = [mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=20),
                mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=20),
                mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=20)]
    labels = ['Original\nmatrix\ncentroids', 'Original\ncluster\naverages', 'New data\ncluster\naverages']
    ax.legend(handles, labels, bbox_to_anchor=(0.99,0.1), loc="lower left")
    plt.show()

def new_abs_cluster_summary_bar_plot (data, new_abstracts, row_col_labels, row_col_centroids, orig_model, bar_plot=True, orig_word_reps=None, logger=None):
    model = FooClass()
    model.__original_data = new_abstracts
    model.row_labels_, model.column_labels_ = row_col_labels
    model.new_centroids = row_col_centroids
    model.__vectorization, model.cluster_assoc =  orig_model.__vectorization, orig_model.cluster_assoc # reuse vectorization and cluster assoc
    
    model.centroids = orig_model.centroids
    if not CLUSTER_CENTER_IS_AVERAGE:
        model.basis_vectors = orig_model.basis_vectors
    
    if hasattr(orig_model,"row_pca"):
        model.col_pca = PCA(n_components=2, random_state=42).fit(normalize(np.vstack([data.T, row_col_centroids[1].T])))
        model.row_pca, model.row_c_palette, model.col_c_palette = orig_model.row_pca, orig_model.row_c_palette, orig_model.col_c_palette

    _, w_occurrence_per_d_cluster = cluster_summary(data, model, 
                                            word_reps=orig_word_reps, logger=logger)
    if bar_plot:
        cocluster_words_bar_plot(w_occurrence_per_d_cluster, filename_suffix="new_abs")

def highlight_passages (new_abstracts, new_abs_classification, row_centroids, orig_vec):
    # TODO: account for non-selected clusters
    m, k = row_centroids.shape
    rex = re.compile("\.\s+")
    pink_text = colored.fg("deep_pink_3b")
    segmented_abstracts = []
    abs_sizes = []
    assert len(new_abstracts) == len(new_abs_classification), "OH NO"

    # segment the abstracts
    for ab in new_abstracts:
        segm_abs = re.split(rex, ab)
        for a,seg in enumerate(segm_abs[:]): # dirty fix
            if a == len(segm_abs)-1 and ("copyright" in seg or "ltd" in seg or "licence" in seg or "license" in seg or "authors" in seg or "society" in seg):
                segm_abs.remove(seg)

        segmented_abstracts.extend(segm_abs)
        abs_sizes.append(len(segm_abs))
    print("len segm", len(segmented_abstracts), "sum", sum(abs_sizes))
    print("zero size abs", np.sum(np.array(abs_sizes) == 0))

    # vectorize each segment
    segm_matrix = orig_vec.transform(segmented_abstracts)
    print("zero segms", np.sum(np.sum(segm_matrix,axis=1) == 0))

    # calculate distances to centroids and select segments with distance smaller than threshold
    segm_count = 0
    all_distances = np.zeros((segm_matrix.shape[0], k), dtype=np.float64)
    big_thing = m
    for i, ab in enumerate(new_abstracts):
        label = new_abs_classification[i]
        N_select = math.ceil(0.2*abs_sizes[i])
        abs_matrix = segm_matrix[segm_count:segm_count+abs_sizes[i]]

        for j,r in enumerate(abs_matrix):
            if not (np.sum(r) == 0):
                all_distances[segm_count+j] = [norm(r-centroid) for centroid in row_centroids.T]
            else:
                all_distances[segm_count+j] = big_thing
        abs_distances = all_distances[segm_count:segm_count+abs_sizes[i], :]

        selected = [segm_idx for dist,segm_idx in sorted(zip(abs_distances[:, label],range(abs_distances.shape[0])))[:N_select]]
        """
        cluster_column = abs_distances[:, label]
        threshold = np.mean(cluster_column[cluster_column != big_thing])
        selected = np.argwhere(abs_distances[:, label] <= threshold) # indices of selected segments
        if i<2:
            print("threshold", threshold)
        """

        if i < 2:
            print("abs dists:\n", abs_distances,sep="")
        
        # print info
        if i < 10:
            print(f"ABSTRACT {i} ( in cluster {label}):\n")
            for n_seg, segm in enumerate(segmented_abstracts[segm_count:segm_count+abs_sizes[i]]):
                if n_seg not in selected:
                    print(f"\t{segm}")
                else:
                    print(f"\t{colored.stylize(segm, pink_text)}")
            print("\n######## ############ ################ ########\n")
        segm_count += abs_sizes[i]


def misc_statistics (orig_model, new_new_abstracts, new_data, row_col_labels): #DBG
    print(f"\nnorm for original centroids (r,c): {norm(orig_model.centroids[0])}, {norm(orig_model.centroids[1])}")
    print(f"mean for original centroids (r,c): {np.mean(orig_model.centroids[0])}, {np.mean(orig_model.centroids[1])}")

    orig_r_clust_avg = get_centroids_by_cluster(orig_model.data, orig_model.row_labels_, orig_model.n_row_clusters)
    new_r_clust_avg = get_centroids_by_cluster(new_data, row_col_labels[0], orig_model.n_row_clusters)
    row_dist = norm(orig_r_clust_avg - new_r_clust_avg, axis=0)
    print(f"Difference between original and new cluster averages:\n\t{row_dist}")

    s1 = set(orig_model.__vectorization.vocabulary_.keys())
    new_vec = TfidfVectorizer(**vec_kwargs)
    new_vec.fit(new_new_abstracts)
    s2 = set(new_vec.vocabulary_.keys())
    print("\n\nvocab1:", len(s1),"vocab2:", len(s2))
    print("vocab difference (2 not in 1):",len(s2.difference(s1)))
    print("tfidf words missing in all new abstracts:", np.sum(np.sum(new_data,axis=0)==0) )

def main():
    global RNG_SEED
    RNG, RNG_SEED = start_default_rng(seed=RNG_SEED)
    np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)

    # read and process
    df = pd.read_csv('data/artigosUtilizados.csv', delimiter=',')
    abstracts = df['abstract']
    new_abstracts = Preprocessor().transform(abstracts)

    # do co-clustering
    # NOTE: docs are normalized (courtesy of sklearn); words arent
    data, vec = do_vectorization(new_abstracts, VECTORIZATION, **vec_kwargs)
    model, statistics = do_task_single(data, new_abstracts, vec, alg=ALG, RNG_SEED=RNG_SEED, show_images=SHOW_IMAGES)
    

    # analyze new abstracts
    if NEW_ABS:
        print("@@##@##@#@#@##@#@#@### #@# @##@ #@ ## @#@# #@ # @##@# @# @#@##@#@#@#@#","\t\tNEW ABSTRACTS\t\t","@@##@##@#@#@##@#@#@### #@# @##@ #@ ## @#@# #@ # @##@# @# @#@##@#@#@#@#", sep="\n")
        new_new_abstracts, df_new_abs = load_new_new_abstracts("data/artigosNaoUtilizados.csv", 496+20, abstracts)
        # get abstract labels (and reuse the word labels)
        Z, new_abs_classification, _ = vec_and_class_new_abstracts(new_new_abstracts, vec, model, verbose=False)
        print_silhouette_score(Z, new_abs_classification, model.column_labels_)

        # calculate new column centroids since we have a new word space (due to having different abstracts)
        new_row_centroids = get_centroids_by_cluster(Z, new_abs_classification, model.n_row_clusters)
        new_col_centroids = get_centroids_by_cluster(Z.T, model.column_labels_, model.n_col_clusters)
        
        # print document and word representatives; do a bar plot summarizing this information
        new_abs_cluster_summary_bar_plot(Z, new_new_abstracts, (new_abs_classification, model.column_labels_), 
            (new_row_centroids, new_col_centroids), model, 
            orig_word_reps=(model.__col_reps if KEEP_WORD_REPS else None), bar_plot=SHOW_IMAGES, logger=None)
        # highlight important passages
        highlight_passages(new_new_abstracts, new_abs_classification, new_row_centroids, vec)

        # distance metrics and info about missing vocabulary, empty words
        misc_statistics(model, new_new_abstracts, Z, (new_abs_classification, model.column_labels_))

        # reduced-dimension scatter plot for new abstracts
        if SHOW_IMAGES:
            new_abs_reduced_centroids_plot(Z, new_abs_classification, model, RNG=RNG)

if __name__ == "__main__":
    main()