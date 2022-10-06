# -*- coding: utf-8 -*-
"""graph_randomization.py

Author  : Matt Rosen
Modified: 1/2019

This file implements utility functions for Ozkan Lab interactome work.

Todo:
    * test exhaustively (main method)

Toconsider:
    * think about other ways of randomizing graph
    * think about other null models
    * test exhaustively (main method)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import scipy.io as sio
import math
import copy
from utilities import save_edgelist

################################################################################
def random_link_removal(adj, pct, save_fn=None):
    """ Remove pct % of links from graph specified by adj.

        Args:    adj     (numpy.ndarray)
                 pct     (float)
                 save_fn (None or str)
        
        Returns: depleted (numpy.ndarray)

    """ 
    
    # do some argument verification
    assert pct >= 0 and pct <= 1, "Argument `pct` should be in range [0, 1]."
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."

    # find links
    links = np.where(adj > 0)

    # choose appropriate pct of links to remove, + remove them
    ind = np.random.choice(len(links[0]), int(pct * len(links[0])), 
            replace=False)
    to_rmv = [links[0][ind], links[1][ind]]
    depleted = copy.deepcopy(adj)
    depleted[to_rmv] = 0

    # save if desired
    if save_fn is not None:
        save_edgelist(depleted, save_fn)

    return depleted

################################################################################
def weight_permutation(adj, save_fn=None):
    """ Permute weights of graph specified by adj.
        
        Args:    adj      (numpy.ndarray)
                 save_fn  (None or str)
        
        Returns: permuted (numpy.ndarray)
    """
    # do some argument verification
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."

    # find links
    links = np.where(adj > 0)

    # permute weights
    permuted = copy.deepcopy(adj)
    r_weights = np.random.permutation(permuted[links])
    permuted[links] = r_weights

    # save if desired
    if save_fn is not None:
        save_edgelist(permuted, save_fn)

    return permuted

################################################################################
def edge_permutation(adj, save_fn=None):
    """ Permute edges of graph specified by adj.
        
        Args:    adj      (numpy.ndarray)
                 save_fn  (None or str)
        
        Returns: permuted (numpy.ndarray)
    """

    # do some argument verification
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."

    # find links
    links = np.where(adj > 0)

    # randomly link network
    new_links = np.random.choice(range(len(adj.flatten())), len(links[0]), replace=False)
    weights = adj[links]
    permuted = np.zeros(len(adj.flatten()))
    permuted[new_links] = weights
    permuted = np.reshape(permuted, adj.shape)

    # save if desired
    if save_fn is not None:
        save_edgelist(permuted, save_fn)

    return permuted

################################################################################
def degree_preserving_randomization(adj, n_swaps = int(1e4), save_fn=None):
    """ Perform degree-preserving randomization of graph
        specified by adj; does this by swapping link pairs
        if they form new link pairs that don't already exist
        in the network structure.
        
        Args:    adj      (numpy.ndarray)
                 save_fn  (None or str)
        
        Returns: permuted (numpy.ndarray)
    """

    # do some argument verification
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert type(n_swaps) == int, "Argument `n_swaps` should be int."

    # find links
    links = np.where(adj > 0)
    weights = adj[links]
    permuted = np.zeros(adj.shape)

    link_set = set(zip(links[0], links[1]))

    # swap link pairs
    swapped = 0
    while swapped < n_swaps:
        l1, l2 = np.random.choice(range(len(links[0])), 2)

        # if either of the generated link pairs exists, continue
        if ((links[0][l1], links[1][l2]) in link_set or 
            (links[0][l2], links[1][l1]) in link_set):
            continue

        # otherwise, add new edges to link set, remove old ones,
        # and update links arrays accordingly
        link_set.remove((links[0][l1], links[1][l1]))
        link_set.remove((links[0][l2], links[1][l2]))
        link_set.add((links[0][l1], links[1][l2]))
        link_set.add((links[0][l2], links[1][l1]))
        l1_targ = links[1][l1]
        l2_targ = links[1][l2]
        links[1][l1] = l2_targ
        links[1][l2] = l1_targ
        swapped += 1


    # add specified connectivity to permuted
    permuted[links] = weights

    # save if desired
    if save_fn is not None:
        save_edgelist(permuted, save_fn)

    return permuted   

################################################################################
