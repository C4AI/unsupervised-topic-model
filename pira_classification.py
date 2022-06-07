#!/usr/bin/env python3


import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.feature_extraction.text import *
import nltk
import os, re
from nbvd import NBVD_coclustering
from my_utils import *
from collections import OrderedDict

try:
    stop_words_nltk=nltk.corpus.stopwords.words('english')
except:
    downloads = [nltk.download('stopwords')]
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

N_ROW_CLUSTERS, N_COL_CLUSTERS = 4,4
RNG_SEED=423
VECTORIZATION='tfidf'
vec_kwargs = Bunch(min_df=4, stop_words=stop_words_nltk, lowercase=False)
ALG='nbvd'
NEW_ABS=True
LOG_BASE_FOLDER = "classification_info"


############################################################################## 
# to use a set number of cpus: 
#   taskset --cpu-list 0-7 python "pira.py"
##############################################################################

exp_numbers = re.compile("[^A-Za-z]\d+\.\d+|[^A-Za-z]\d+\,\d+|[^A-Za-z]\d+")
exp_non_alpha = re.compile("[^A-Za-zÀàÁáÂâÃãÉéÊêÍíÓóÔôÕõÚúÇç02-9 \_–+]+")
exp_whitespace = re.compile("\s+")
exp_hyphen = re.compile("(?<=[a-z])\-(?=[a-z])")

# NOTES: no overly small abstracts (all greater than 300 characters); 
# some duplicates (in the abstract column)
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
            if sentence not in unique_sentences and len(sentence) > 300:
                unique_sentences.add(sentence)
                newX.append(self.preprocess(sentence))
        return newX

def do_vectorization (new_abstracts, vectorization_type, **kwargs):
    if vectorization_type == 'tfidf':
        vec = TfidfVectorizer(**kwargs)
        data = vec.fit_transform(new_abstracts).toarray()
    elif vectorization == 'count':
        vec = CountVectorizer(**kwargs)
        data = vec.fit_transform(new_abstracts).toarray()
    elif vectorization_type == 'tfidf-char':
        vec = TfidfVectorizer(ngram_range=(5,5), analyzer='char')
        data = vec.fit_transform(new_abstracts).toarray()

    return (data, vec)

def do_task_single (data, original_data, vectorization, alg=ALG, 
        RNG_SEED=None, logger=None, iter_max=2000):
    RNG = np.random.default_rng(RNG_SEED)

    if logger:
        logger.info(f"shape: {data.shape}")
    else:
        print(f"shape: {data.shape}")

    # do co-clustering
    if alg == 'nbvd':
        model = NBVD_coclustering(data, n_row_clusters=N_ROW_CLUSTERS, 
            n_col_clusters=N_COL_CLUSTERS, n_attempts=1, iter_max=iter_max, random_state=RNG_SEED, verbose=True)

    #########################
    # evaluate results 
    #########################

    ### internal indices
    # print silhouette scores
    silhouette = print_silhouette_score(data, model.row_labels_, model.column_labels_, logger=logger)

    # return general statistics
    if alg == 'nbvd':
        bunch = Bunch(silhouette=MeanTuple(*silhouette), 
            best_iter=model.best_iter, best_norm=model.best_norm, n_attempts=1)
    return (model, bunch)

def get_article_ids (abstracts, df):
    df = df.reset_index() # make sure indexes are correct
    ids_abs = OrderedDict([(ab,(-1,"")) for ab in abstracts])
    ids_found = 0
    abstracts_s = set(abstracts)
    if len(abstracts_s) != len(abstracts):
        raise Exception("list of abstracts contains duplicates")

    for i, row in df.iterrows():
        processed_ab = Preprocessor().preprocess(row['abstract'])
        if processed_ab in abstracts_s:
            ids_abs[processed_ab] = (row['idarticle'], row['abstract'])
            ids_found += 1
        if ids_found == len(ids_abs):
            break
    return ids_abs.values()

def load_new_new_abstracts (path, n_abstracts, old_abstracts):
    old_abstracts_S = set(old_abstracts)
    df = pd.read_csv(path, delimiter=',')
    new_new_abstracts = df['abstract'][:n_abstracts].to_list()
    new_new_not_repeat = [ab for ab in new_new_abstracts if ab not in old_abstracts_S]
    new_processed_abstracts = Preprocessor().transform(new_new_not_repeat) # preprocess and eliminate duplicates
    print(f"\nnew abstracts: {len(new_processed_abstracts)} | old abstracts present: {len(new_new_abstracts) - len(new_new_not_repeat)}")
    return new_processed_abstracts, df

def vec_and_class_new_abstracts (extra_abstracts : Iterable, vec, model, logger=None):
    print_or_log = logger.info if logger else print
    row_centroids, col_centroids = model.centroids
    m, k = row_centroids.shape
    n, l = col_centroids.shape

    # vectorize abstracts
    Z = vec.transform(extra_abstracts).toarray()
    n, _ = Z.shape

    # classify rows and columns
    row_classification = NBVD_coclustering.get_labels(Z, row_centroids, k, m, n)
    col_classification = model.column_labels_

    return (Z, row_classification, col_classification)

def write_ids_to_excel (abstracts, row_classification, abstracts_df, excel_filename):
    os.makedirs(LOG_BASE_FOLDER, exist_ok=True)
    sheet_path = os.path.join(LOG_BASE_FOLDER, excel_filename)
    writer = pd.ExcelWriter(path=sheet_path, engine='xlsxwriter')
    abstracts = np.array(abstracts)

    for c in range(N_ROW_CLUSTERS):
        sheet = pd.DataFrame(columns=['idarticle', 'abstract']) # create new empty sheet for this cluster

        # get ids
        ids_abs = get_article_ids(abstracts[row_classification == c], abstracts_df)
        ids, unprocessed_abs = zip(*ids_abs)
        sheet['idarticle'] = ids
        sheet['abstract'] = unprocessed_abs
        
        # write to excel file
        sheet_name = f"Cluster {c}"
        sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.sheets[sheet_name].set_column(0, 0, 15)
    writer.save() # else it won't save

def main():
    global RNG_SEED
    RNG, RNG_SEED = start_default_rng(seed=RNG_SEED)

    # read and process
    df = pd.read_csv('data/artigosUtilizados.csv', delimiter=',')
    abstracts = df['abstract']
    new_abstracts = Preprocessor().transform(abstracts)

    # do vectorization and co-clustering
    data, vec = do_vectorization(new_abstracts, VECTORIZATION, **vec_kwargs)
    model, statistics = do_task_single(data, new_abstracts, vec, alg=ALG, iter_max=2000, RNG_SEED=RNG_SEED)
    
    # get article ids
    write_ids_to_excel(new_abstracts, model.row_labels_, df, "orig_abstracts.xlsx")

    # analyze new abstracts
    if NEW_ABS:
        print("@@##@##@#@#@##@#@#@### #@# @##@ #@ ## @#@# #@ # @##@# @# @#@##@#@#@#@#","\t\tNEW ABSTRACTS\t\t","@@##@##@#@#@##@#@#@### #@# @##@ #@ ## @#@# #@ # @##@# @# @#@##@#@#@#@#", sep="\n")
        new_new_abstracts, new_abstracts_df = load_new_new_abstracts("data/artigosNaoUtilizados.csv", 496+20, abstracts)
        Z, new_abs_classification, _ = vec_and_class_new_abstracts(new_new_abstracts, vec, model)
        print_silhouette_score(Z, new_abs_classification, model.column_labels_)
        write_ids_to_excel(new_new_abstracts, new_abs_classification, new_abstracts_df, "new_abstracts.xlsx")

if __name__ == "__main__":
    main()