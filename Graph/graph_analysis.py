# -*- coding: utf-8 -*-
"""graph_analysis.py

Author  : Matt Rosen
Modified: 1/2019

This file implements graph analysis functions for Ozkan Lab interactome work.

Todo:
    * Implement degree computation
    * Implement degree distribution computation
    * Implement centrality computations
    * Implement clustering functions

Toconsider:
    * Questions about predicting which nodes should merge/
      e.g. splits backward in time. Why should a node split?

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import scipy.io as sio
import math
import copy
import networkx as nx
from utilities import *

################################################################################
def compute_centrality_scores(adj, methods, names, save_fn=None):
    """ Compute centrality using provided NetworkX functions.

        Args:    adj          (numpy.ndarray)
                 methods      (list)
                 names        (list of str)
                 save_fn      (None or str)
              
        Returns: centralities (dict of dicts)
        
    """

    # some argument checking
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert type(methods) == list, "Argument `methods` should be a list."
    assert type(names) == list and type(names[0]) == str, \
        "Argument `names` should be a list of str."
    assert save_fn is None or type(save_fn) == str

    # form graph from adj
    G = nx.from_numpy_array(adj)
    
    centralities = {}
    
    # compute centrality scores based on methods supplied
    for name, method in zip(names, methods):
        centralities[name] = method(G)
    
    return centralities

################################################################################
def compute_community_assignments(adj, methods, names, args, save_fn=None):
    """ Compute community assignments using provided NetworkX functions.

        Args:     adj              (numpy.ndarray)
                  methods          (list of fxns)
                  names            (list of str)
                  args             (list of arg dicts)
                  save_fn          (None or str)
              
        Returns:  comm_assignments (dict of dicts)
        
    """

    # some argument checking
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert type(methods) == list, "Argument `methods` should be a list."
    assert type(names) == list and type(names[0]) == str, \
        "Argument `names` should be a list of str."
    assert save_fn is None or type(save_fn) == str

    # form graph from adj
    G = nx.from_numpy_array(adj)
    
    comm_assignments = {}
    
    # compute comm. assignments based on methods supplied
    for name, method, arg in zip(names, methods, args):
        comm_assignments[name] = method(G, **arg)
    
    return comm_assignments

################################################################################
def compute_degrees(adj, save_fn=None):
    """ Compute node degrees for provided graph.

        Args:    adj     (numpy.ndarray)
                 save_fn (None or str)
        
        Returns: degrees (dict)
        
    """

    # some argument checking
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert save_fn is None or type(save_fn) == str

    # form graph from adj
    G = nx.from_numpy_array(adj)
    
    # compute + return node degrees
    return dict(G.degree())
