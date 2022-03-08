#!/usr/bin/env python3

import numpy as np
from numpy.linalg import inv, norm
from numpy.random import default_rng
from matplotlib import pyplot as plt
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering, KMeans
from sklearn.metrics import silhouette_score, consensus_score, accuracy_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import *
from sklearn.preprocessing import normalize
from collections import Counter
import itertools
from pprint import pprint
import sys,os,pickle,re
import algorithms
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl
import pyqtgraph as pg

wbkm = __import__("wbkm numpy debugging land")
from nbvd import NBVD_coclustering
from my_utils import *

np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)

# w2v:
# 42123456 (8,5)
# 12 (4,6) dim=50
N_ROW_CLUSTERS, N_COL_CLUSTERS = 4,4
RNG_SEED=42
VECTORIZATION='tfidf'
W2V_DIM=100
ALG='nbvd'
ATTEMPTS_MAX=1
SYMMETRIC = False # (NBVD) use symmetric NBVD algorithm?
WAIT_TIME = 4 # wait time between tasks
SHOW_IMAGES = True # display matrices
LABEL_CHECK = True
CENTROID_CHECK = True
COCLUSTER_CHECK = True
NORM_PLOT = False # (NBVD) display norm plot
CENTROID_REPRESENTATIVE=False
rerun_embedding=True
MOVIE=False
ASPECT_RATIO=4 # 1/6 for w2v; 10 for full tfidf



############################################################################## 
# to use a set number of cpus: 
#   taskset --cpu-list 0-7 python "pira.py"
##############################################################################

def w2v_combine_sentences (tokenized_sentences, model, method='tfidf', isDoc2Vec=False):
    out = []
    if method == 'tfidf' and not isDoc2Vec:
        pseudo_sentences = [" ".join(token_list) for token_list in tokenized_sentences]
        vec = TfidfVectorizer()
        vec.fit(pseudo_sentences)
        idf_mapping = dict(zip(vec.get_feature_names(), vec.idf_))
    elif method == 'concat':
        max_len = max([len(sen) for sen in tokenized_sentences])

    for i,sentence in enumerate(tokenized_sentences):
        if not isDoc2Vec:
            if model.isPreTrained:
                word_vectors = np.array([model.word_to_vec_dict[word] for word in sentence if word in model.word_to_vec_dict])
            else:
                word_vectors = np.array([model.wv[model.wv.key_to_index[word]] for word in sentence])
            if method == 'sum':
                sentence_vector = np.sum(word_vectors, axis=0)
            elif method == 'mean':
                sentence_vector = np.mean(word_vectors, axis=0)
            elif method == 'tfidf':
                if not model.isPreTrained:
                    word_idfs = [idf_mapping[word] for word in sentence]
                else:
                    word_idfs = [idf_mapping[word] for word in sentence if word in model.word_to_vec_dict]
                n = word_vectors.shape[0]
                sentence_vector = np.dot(word_idfs, word_vectors)/n # average with idf as weights
            elif method == 'concat':
                #NOTE: maybe get normed vectors above and use normed vectors? 
                # normalize sentence_vector?
                flattened = word_vectors.flatten()
                pad_length = W2V_DIM*max_len - flattened.shape[0]
                # pad 'flattened' with constant zeroes, 
                #   'pad_length' times _after_ the original array and 0 times _before_ it
                #   (we're doing post-padding)
                sentence_vector = np.pad(flattened, (0, pad_length), 'constant', constant_values=0)
        else:
            sentence_vector = model.dv[i]
        out.append(sentence_vector)
    out = np.array(out)
    if method == 'concat':
        out = normalize(np.array(out), norm='l1', copy=False) #l2 didnt do much good
    return out

def dict_from_pretrained_model(path_to_embedding_file, relevant_words):
    print("Creating embedding matrix ...")
    embeddings_index = {}
    with open(path_to_embedding_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                n_vectors, embedding_dim = line.split()
                continue
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype="float64", sep=" ")
            if word in relevant_words:
                embeddings_index[word] = coefs
    misses = relevant_words.difference(set(embeddings_index.keys()))
    print(f"{len(misses)} misses while converting")
    return embeddings_index

class FullModel:
    pass
exp_numbers = re.compile("\d+")
exp_non_alpha = re.compile("[^A-Za-zÀàÁáÂâÃãÉéÊêÍíÓóÔôÕõÚúÇç1 -]+")
exp_whitespace = re.compile("\s+")

# no overly small abstracts (all greater than 300 characters); 
# some duplicates (in the abstract column)
class Preprocessor:
    def preprocess (self, sentence):
        new_sentence = sentence
        new_sentence = re.sub(exp_numbers, "1", new_sentence)
        new_sentence = re.sub(exp_non_alpha, " ", new_sentence)
        new_sentence = re.sub(exp_whitespace, " ", new_sentence)
        return new_sentence.lower()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_list = X.to_list() if type(X) != list else X
        unique_sentences = set()
        newX = []
        for i,sentence in enumerate(X_list):
            if sentence not in unique_sentences:
                unique_sentences.add(sentence)
                newX.append(self.preprocess(sentence))
        return newX

def get_representatives (data, labels, centroids, n_clusters, n_representatives=3):
    cluster_representatives = {}
    print("picking representatives:")
    if CENTROID_REPRESENTATIVE:
        all_distances = np.zeros((data.shape[0], n_clusters))
        for i, r in enumerate(data):
            all_distances[i] = [norm(r-centroid) for centroid in centroids.T]
        for c in range(n_clusters):
            dists_to_centroid = sorted(zip(list(range(data.shape[0])), list(all_distances[:,c])), 
                key = lambda t : t[1])

            # select top n; eliminate candidates that arent from the relevant cluster
            rep_candidates = [t[0] for t in dists_to_centroid[:n_representatives]]
            print(len(rep_candidates), end=" and then ")
            for cand in rep_candidates[:]:
                if labels[cand] != c:
                    rep_candidates.remove(cand)
            print(len(rep_candidates))
            if rep_candidates: # if anyones left
                cluster_representatives[c] = rep_candidates
    else:
        # get cluster averages
        cluster_avgs = {}
        for r, label in enumerate(labels):
            if label not in cluster_avgs:
                cluster_avgs[label] = []
            cluster_avgs[label].append(data[r])
        for k in cluster_avgs.keys():
            cluster_avgs[k] = np.sum(cluster_avgs[k]) / len(cluster_avgs[k])

        ord_clusters = [t[1] for t in sorted(cluster_avgs.items())] # len might be ower than n_clusters
        all_distances = np.zeros((data.shape[0], len(ord_clusters)))
        for i, r in enumerate(data):
            all_distances[i] = [norm(r-cluster_avg) for cluster_avg in ord_clusters]
        for c in range(len(ord_clusters)):
            dists_to_centroid = sorted(zip(list(range(data.shape[0])), list(all_distances[:,c])), 
                key = lambda t : t[1])
                
            # select top n; eliminate candidates that arent from the relevant cluster
            rep_candidates = [t[0] for t in dists_to_centroid[:n_representatives]]
            print(len(rep_candidates), end=" and then ")
            for cand in rep_candidates[:]:
                if labels[cand] != c:
                    rep_candidates.remove(cand)
            print(len(rep_candidates))
            if rep_candidates: # if anyones left
                cluster_representatives[c] = rep_candidates
    return cluster_representatives

def cluster_summary (data, original_data, vec, model, logger=None):
    print_or_log = logger.info if logger else print
    cluster_assoc = model.cluster_assoc
    assoc_shape = cluster_assoc.shape
    row_centroids, col_centroids = model.centroids

    """
    # get relevant coclusters
    relevant_coclusters = []
    for i in range(assoc_shape[0]):
        for j in range(assoc_shape[1]):
            if cluster_assoc[i,j]:
                relevant_coclusters.append((i,j))
    """
    
    # get relevant documents
    row_cluster_representatives = get_representatives(data, model.row_labels_, row_centroids, 
        assoc_shape[0], n_representatives=3)
    col_cluster_representatives = get_representatives(data.T, model.column_labels_, col_centroids, 
        assoc_shape[1], n_representatives=10)

    ## TODO: check for relevant cocluster?
    # documents
    print_or_log("DOCUMENTS:\n")
    for i, reps in sorted(row_cluster_representatives.items()):
        print_or_log("cluster:", i)
        for rep in reps:
            print_or_log("rep:", rep)
            print_or_log(original_data[rep][:200])
        print_or_log("--------------------------------------------------------\n")
    print("stuffs")
    print(model.row_labels_[119])
    print(vec.vocabulary_["to"], vec.vocabulary_["for"], vec.vocabulary_["with"])
    print(model.column_labels_[vec.vocabulary_["to"]], model.column_labels_[vec.vocabulary_["for"]], 
        model.column_labels_[vec.vocabulary_["with"]])

    # words
    print_or_log("WORDS:\n")
    if isinstance(vec, TfidfVectorizer):
        idx_to_word = vec.get_feature_names()
        for i, c in sorted(col_cluster_representatives.items()):
            print_or_log("cluster:", i)
            to_print = []
            for rep in c:
                to_print.append(idx_to_word[rep])
            print_or_log(*to_print, sep=", ")
            print_or_log("--------------------------------------------------------\n")

def do_task_single (data, original_data, vectorization, only_one=True, alg=ALG, n_attempts=ATTEMPTS_MAX, 
        show_images=True, first_image_save_path=None, RNG_SEED=None, logger=None):
    RNG = np.random.default_rng(RNG_SEED)
    if logger:
        logger.info(f"shape: {data.shape}")
    else:
        print(f"shape: {data.shape}")
    timer = None if only_one else WAIT_TIME * show_images 

    """
    if not only_one:
        # plot original data to build suspense AND save figure if a save path is provided
        plot_matrices([data], ["Original dataset"], timer=timer, savefig=first_image_save_path)
    """

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

    ### internal indices
    # print silhouette scores
    silhouette = print_silhouette_score(data, model.row_labels_, model.column_labels_, logger=logger)

    # textual analysis
    cluster_summary (data, original_data, vectorization, model, logger=None)

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
            shaded_label_matrix(data, model.row_labels_, kind="rows",method_name="rows", RNG=RNG, opacity=1, aspect_ratio=ASPECT_RATIO)
            shaded_label_matrix(data, model.column_labels_, kind="columns",method_name="columns", RNG=RNG, opacity=1, aspect_ratio=ASPECT_RATIO)
            if COCLUSTER_CHECK and alg == "nbvd":
                shade_coclusters(data, (model.row_labels_, model.column_labels_), 
                    model.cluster_assoc, RNG=RNG, aspect_ratio=ASPECT_RATIO)
             
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
        plot_matrices(to_plot, names, timer = None if only_one else 2*timer, aspect_ratio=ASPECT_RATIO)

    """
    # textual analysis
    cluster_summary (data, original_data, vec, model, logger=None)"""

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

def main():
    global RNG_SEED
    RNG, RNG_SEED = start_default_rng(seed=RNG_SEED)
    np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)

    #read
    df = pd.read_csv('pira.csv', delimiter=';')
    scientific = df[df['corpus'] == 1]
    abstracts = scientific['abstract']
    new_abstracts = Preprocessor().transform(abstracts)

    os.makedirs('.embedding_cache', exist_ok=True)
    if VECTORIZATION == 'tfidf':
        # DBG: possibly dont restrict vocabulary?
        #vec = TfidfVectorizer()
        vec = TfidfVectorizer(min_df=4, max_df=0.98, max_features=2000)
        data = vec.fit_transform(new_abstracts).toarray()
    elif VECTORIZATION == 'tfidf-char':
        vec = TfidfVectorizer(ngram_range=(5,5), analyzer='char')
        data = vec.fit_transform(new_abstracts).toarray()
    elif VECTORIZATION == 'w2v' or VECTORIZATION == 'pretrained':
        embedding_dump_name = ".embedding_cache/pira.w2v" if VECTORIZATION == 'w2v' else ".embedding_cache/pretrained.w2v"
        vec = CountVectorizer()
        tok = vec.build_tokenizer()
        tokenize_sentence_array = lambda sentences: [tok(sentence) for sentence in sentences]
        tok_sentences = tokenize_sentence_array(new_abstracts) # frases tokenizadas (iteravel de iteraveis)
        if rerun_embedding or not os.path.isfile(embedding_dump_name):
            if VECTORIZATION == 'w2v':
                # NOTE: min_count=1 is not the default >:) exhilarating!
                full_model = Word2Vec(sentences=tok_sentences, vector_size=W2V_DIM, min_count=1, sg=0, window=5, workers=1, seed=42) # ligeiramente melhor mas mt pouco
                full_model.save(embedding_dump_name)
            elif VECTORIZATION == 'pretrained':
                vec.fit(new_abstracts)
                full_model = FullModel() # lambdas cannot be pickled apparently
                full_model.isPreTrained = True
                full_model.vocabulary = set(vec.vocabulary_.keys())
                full_model.word_to_vec_dict = dict_from_pretrained_model(path_to_embedding_file="pre-trained/cc.en.300.vec", relevant_words=full_model.vocabulary)
                with open(embedding_dump_name, "wb") as f:
                    pickle.dump(full_model, f)
        else:
            if VECTORIZATION == 'w2v':
                full_model = Word2Vec.load(embedding_dump_name, mmap='r')
            elif VECTORIZATION == 'pretrained':
                with open(embedding_dump_name, "rb") as f:
                    full_model = pickle.load(f)
        if VECTORIZATION == 'w2v':
            full_model.isPreTrained = False
        vec = full_model
        data = w2v_combine_sentences(tok_sentences, full_model, method='tfidf')
        
    elif VECTORIZATION == 'd2v':
        embedding_dump_name = ".embedding_cache/pira.d2v"
        tok = CountVectorizer().build_tokenizer()
        tokenize_sentence_array = lambda sentences: [tok(sentence) for sentence in sentences]
        tok_sentences = tokenize_sentence_array(new_abstracts) # frases tokenizadas (iteravel de iteraveis)
        if rerun_embedding or not os.path.isfile(embedding_dump_name):
            # NOTE: min_count=1 is not the default >:) exhilarating!
            documents = [TaggedDocument(sentence, [i]) for i,sentence in enumerate(tok_sentences)]
            doc_vector_size = 1*W2V_DIM
            full_model = Doc2Vec(documents=documents, vector_size=doc_vector_size, min_count=1, dm=0, window=5, workers=1, seed=42) # ligeiramente melhor mas mt pouco
            full_model.save(embedding_dump_name)
        else:
            full_model = Doc2Vec.load(embedding_dump_name, mmap='r')
        vec = full_model
        data = w2v_combine_sentences(tok_sentences, full_model, isDoc2Vec=True)

    print("shape, type:", data.shape, type(data))
    # do co-clustering
    results = do_task_single(data, new_abstracts, vec, alg=ALG, RNG_SEED=RNG_SEED)




















    """
    # do co-clustering
    if ALG == 'nbvd':
        model = NBVD_coclustering(data, n_row_clusters=N_ROW_CLUSTERS, n_col_clusters=N_COL_CLUSTERS, 
            n_attempts=ATTEMPTS_MAX, random_state=RNG_SEED, 
            verbose=True, save_history=MOVIE)
    elif ALG == 'wbkm':
        model = wbkm.WBKM_coclustering(data, N_ROW_CLUSTERS, random_state=RNG_SEED, 
        n_attempts=ATTEMPTS_MAX, verbose=True)
    elif ALG == 'spectral':
        model = SpectralCoclustering(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        model.fit(data)
    elif ALG == 'kmeans':
        model = lambda : None
        # does this make sense i dont know
        row_model = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        row_model.fit(data)
        model.row_labels_ = row_model.labels_
        col_model = KMeans(n_clusters=N_ROW_CLUSTERS, random_state=RNG_SEED)
        col_model.fit(data.T)
        model.column_labels_ = col_model.labels_
    elif ALG == 'nbvd_waldyr':
        U, S, V, resNEW, itr = algorithms.NBVD(data, N_ROW_CLUSTERS, N_COL_CLUSTERS, itrMAX=2000)
        model = lambda: None
        model.biclusters_, model.row_labels_, model.column_labels_ = NBVD_coclustering.get_stuff(U, V.T)
        print("resNEW, itr:", resNEW, itr) # resNEW is norm squared; itr is iteration_no

    # internal indices
    print_silhouette_score(data, model.row_labels_, model.column_labels_)
    
    ##################################################### 
    # pyqtgraph
    #####################################################
    if MOVIE and ALG == 'nbvd':
        pyqtgraph_thing (data, model, 25)
    """
if __name__ == "__main__":
    main()