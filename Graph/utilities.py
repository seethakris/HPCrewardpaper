# -*- coding: utf-8 -*-
"""utilities.py

Author  : Matt Rosen
Modified: 1/2019

This file implements utility functions for Ozkan Lab interactome work.

Todo:
    * Finish writing threshold_transform() and sparsity_transform()
    * Test exhaustively (through main method)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import scipy.io as sio
import warnings
import math
import copy
import os

################################################################################
def save_edgelist(adj, save_fn, aggressive_compress=True):
    """ Save graph specified by adj to filename save_fn;
        if aggressive_compress, split columns, write separately,
        compress separately. 

        Args:    adj                 (numpy.ndarray)
                 save_fn             (str)
                 aggressive_compress (boolean)

        Returns: 0
    """
    # select links
    edges    = np.where(adj > 0)
    weights  = adj[adj > 0]
    to_write = np.array([edges[0], edges[1], weights])

    # add .gz to the end of save_fn if not already there (no reason not to)
    if not save_fn.endswith(".gz"):
        save_fn += ".gz"

    # make sure folder exists; if not, make it
    if not os.path.isdir(save_fn[:save_fn.rfind('/')]):
        os.makedirs(save_fn[:save_fn.rfind('/')])
        
    # split columns, write them to separate files; delta encoding for first col
    # should save some space.
    if aggressive_compress:

        # delta encoding for src edge (takes advantage of sorting)
        e0         = edges[0][0]
        xs         = np.insert(np.diff(edges[0]), e0, 0)
        e_to_write = np.array([xs.astype(np.uint8), edges[1]])

        # new filenames
        edges_fn   = save_fn[:-7] + "_edges.txt.gz"
        weights_fn = save_fn[:-7] + "_weights.txt.gz"

        # write edges + values separately
        np.savetxt(edges_fn, e_to_write.T, fmt='%d %d')
        np.savetxt(weights_fn, weights.T, fmt='%.3f')

    else:
        np.savetxt(save_fn, to_write.T, fmt='%d %d %1.3f')

    return 0

################################################################################
def check_triangular(adj, sparsity=0.01):
    """ Check if matrix adj is triangular; make upper-triangular + warn, if 
        not.

        Args:    adj        (numpy.ndarray)
        
        Returns: triangular (numpy.ndarray)
    """

    # check arg first
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert adj.shape[0] == adj.shape[1], "Argument `adj` should be square."

    # if adj is upper-triangular, return
    if np.allclose(adj, np.triu(adj)):
        return

    # if it's lower triangular, flip it, return
    elif np.allclose(adj, np.tril(adj)):
        return adj.T

    # if it's neither, warn, take max of matched weights
    else:
        warnings.warn("Matrix provided to check_triangular() "
            + "is not triangular. Taking max of (i, j) and (j, i) and "
            + f"enforcing default sparsity ({str(sparsity)})");
        lt = np.tril(adj).T
        ut = np.triu(adj)
        maxed = np.amax(np.stack([lt, ut], axis=0), axis=0)

        # enforce sparsity, return 
        threshold_weight = np.percentile(adj.flatten(), 1 - sparsity)
        triangular = copy.deepcopy(adj)
        triangular[triangular < threshold_weight] = 0

        return triangular

################################################################################
def threshold_transform(adj, old_thresh, new_thresh):
    """ Take adjacency matrix created using threshold old_thresh
        and transform into adjacency matrix thresholded by
        new_thresh.

        Args:    adj         (numpy.ndarray)
                 old_thresh  (float)
                 new_thresh  (float)
        
        Returns: thresh_new (numpy.ndarray)
    """

    # check arg first
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert adj.shape[0] == adj.shape[1], "Argument `adj` should be square."

    # take old adjacency matrix, apply new threshold

    return

################################################################################
def sparsity_transform(adj, old_pct, new_pct):
    """ Take adjacency matrix created using sparsity pct old_pct
        and transform into adjacency matrix w/ sparsity pct
        new_pct.

        Args:    adj          (numpy.ndarray)
                 old_pct      (float)
                 new_pct      (float)
        
        Returns: sparsity_new (numpy.ndarray)
    """

    # check arg first
    assert type(adj) == np.ndarray, "Argument `adj` should be numpy.ndarray."
    assert len(adj.shape) == 2, "Argument `adj` should be 2d."
    assert adj.shape[0] == adj.shape[1], "Argument `adj` should be square."

    # take old adjacency matrix, apply new threshold

    return

