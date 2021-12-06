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
from my_utils import MeanTuple, cool_header_thing, plot_matrices, start_default_rng, print_silhouette_score, bic_boolean_to_labels, pyqtgraph_thing

np.set_printoptions(edgeitems=5, threshold=sys.maxsize,linewidth=95) # very personal preferences :)

N_ROW_CLUSTERS, N_COL_CLUSTERS = 4,4
RNG_SEED=42
VECTORIZATION='w2v'
W2V_DIM=100
ALG='nbvd'
ATTEMPTS_MAX=10
rerun_embedding=True
MOVIE=True

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
class Preprocessor:
    minimum_length = 500
    def preprocess (self, sentence):
    # maybe inverting the order would be nice
        new_sentence = sentence
        new_sentence = re.sub(exp_numbers, "1", new_sentence)
        new_sentence = re.sub(exp_non_alpha, " ", new_sentence)
        new_sentence = re.sub(exp_whitespace, " ", new_sentence)
        return new_sentence.lower()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_list = X.to_list() if type(X) != list else X
        newX = []
        for i,sentence in enumerate(X_list):
            # exclude really small texts, such as:
            # '© 2020'
            if len(sentence) > Preprocessor().minimum_length:
                newX.append(self.preprocess(sentence))
        return newX

def main():
    #read
    df = pd.read_csv('pira.csv', delimiter=';')
    scientific = df[df['corpus'] == 1]
    abstracts = scientific['abstract']
    new_abstracts = Preprocessor().transform(abstracts)

    os.makedirs('.embedding_cache', exist_ok=True)
    if VECTORIZATION == 'tfidf':
        vec = TfidfVectorizer()
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
        data = w2v_combine_sentences(tok_sentences, full_model, isDoc2Vec=True)

    print("shape, type:", data.shape, type(data))

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

if __name__ == "__main__":
    main()