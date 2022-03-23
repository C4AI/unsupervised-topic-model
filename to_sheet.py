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
from my_utils import *

def read_results(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    bunches, current_bunch = [], Bunch()
    for line in lines:
        if not line.strip() or "seed" in line or "options" in line:
            continue
        spl = line.split(sep=":", maxsplit=1)
        if len(spl) != 2:
            continue
        
        name, value = spl
        value = value.strip()
        if name == "normal" or name == "binário":
            value1, mean = value.split("| média:")
            value1, mean = value1.strip(), mean.strip()
            current_bunch[name] = value1
            current_bunch[name+"_media"] = f"{float(mean):.4f}"
            if name == "normal":
                mean_normal = float(mean)
            elif name == "binário":
                maior = "Não"
                if float(mean) > mean_normal:
                    maior = "Sim"
                current_bunch["binario_maior"] = maior
                bunches.append(current_bunch)
                current_bunch = Bunch()
        else:
            current_bunch[name] = value
        
    return bunches
from random import random
def fill_sheet (sheet, results, task_name):
    pretty = {
    'vectorizer': 'Vetorizador',
    'vec_kwargs': 'Opções',
    'iter_max': '(NBVD) iterações',
    'shape': 'Formato',
    'normal': 'Silhouette matriz normal',
    'normal_media': 'Média silhouette normal',
    'binário_media': 'Média silhouette binária',
    'binário': 'Silhouette matriz binária',
    'binario_maior': 'Binária é melhor?'
    }
    series = pd.Series(name=task_name, dtype=object) # series name will be the row name
    if results:
        for name, value in results.items():
            series[pretty[name]] = value
    return sheet.append(series)

sheet_path = os.path.join('.', 'binario_ou_nao.xlsx')
#writer = pd.ExcelWriter(path=sheet_path) # openpyxl by default
writer = pd.ExcelWriter(path=sheet_path, engine='xlsxwriter')
sheet = pd.DataFrame()
results = read_results("log_shutdown.txt")

for bunch in results:
    sheet = fill_sheet(sheet, bunch, task_name='')
sheet.to_excel(writer, sheet_name='binario_ou_nao')
#adjust_column_width(sheet, writer, sheet_name='binario_ou_nao')

writer.save() # dont forget!
