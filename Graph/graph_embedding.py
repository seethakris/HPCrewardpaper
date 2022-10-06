# -*- coding: utf-8 -*-
"""graph_embedding.py

Author  : Matt Rosen
Modified: 1/2019

This file implements graph embedding functions for Ozkan Lab interactome work,
using GEM library (Goyal & Ferrara, 2018).

Todo:
    * Figure out how to deal with hyperparameters.
    * TEST these

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import scipy.io as sio
import math
import copy
from utilities import *

# embedding-specific imports
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from time import time

from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE

################################################################################
def embed_graph(adj, methods=['all'], verbose=False):
    """ Embed graph specified by adj; return numpy array 
        of embedding coords.

        Args:   adj         (numpy.ndarray)
                methods     (list of str)
                d           (int)

        Returns: embeddings (numpy.ndarray)

    """

    # some argument checking
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert type(methods) == list and type(methods[0]) == str, 
        "Argument `methods` should be list of str."
    assert type(d) == int, "Argument `d` should be int."
    assert d <= min(adj.shape), "Argument `d` (embedding dimension) should " +
        "be < min(adj.shape)"

    # transform graph into networkx format
    G = nx.from_numpy_matrix(adj)
    G = G.to_directed()

    # set up models
    all_in = 'all' in methods
    models = []

    # d       : embedding dimension
    # max_iter: maximum iterations 
    # eta     : learning rate
    # regu    : regularization coefficient
    if all_in or 'GraphFactorization' in methods:
        models.append(GraphFactorization(d=2, 
                                         max_iter=100000, 
                                         eta=1e-4, 
                                         regu=1.0))

    # d   : embedding dimension
    # beta: decay factor
    if all_in or "HOPE" in methods:
        models.append(HOPE(d=4, beta=0.01))

    # d: embedding dimension
    if all_in or "LaplacianEigenmaps" in methods:
        models.append(LaplacianEigenmaps(d=2))

    # d: embedding dimension
    if all_in or "LocallyLinearEmbedding" in methods:
        models.append(LocallyLinearEmbedding(d=2))

    # d        : embedding dimension
    # max_iter : maximum iteration
    # walk_len : random walk length
    # num_walks: number of random walks
    # con_size : context size
    # ret_p    : return weight 
    # inout_p  : inout weight
    if all_in or "node2vec" in methods:
        models.append(node2vec(d=2, 
                               max_iter=1, 
                               walk_len=80, 
                               num_walks=10, 
                               con_size=10, 
                               ret_p=1, 
                               inout_p=1))

    # d         : embedding dimension
    # beta      : seen edge reconstruction weight
    # alpha     : first order proximity weight
    # nu1       : lasso regularization coefficient
    # nu2       : ridge regression coefficient
    # K         : number of hidden layers
    # n_units   : size of each layer
    # n_ite     : number of iterations
    # xeta      : learning rate 
    # n_batch   : size of batch 
    # modelfile : location of modelfile save 
    # weightfile: location of weightfile save
    if all_in or "SDNE" in methods:
        models.append(SDNE(d=2, 
                           beta=5, 
                           alpha=1e-5, 
                           nu1=1e-6, 
                           nu2=1e-6, 
                           K=3, 
                           n_units=[50, 15,], 
                           n_iter=50, 
                           xeta=0.01, 
                           n_batch=500,
                           modelfile=['enc_model.json', 'dec_model.json'],
                           weightfile=['enc_weights.hdf5', 'dec_weights.hdf5']))

    # perform the embeddings
    for embedding in models:
        if verbose:
            print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()

        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, 
                                         edge_f=None, 
                                         is_weighted=True, 
                                         no_python=True)
        if verbose:
            print(embedding._method_name+':\n\tTraining time: %f' % (time() - t1))

        # Evaluate on graph reconstruction
        MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        
        if verbose:
            print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))

    return