# -*- coding: utf-8 -*-
"""graph_generation.py

Author  : Matt Rosen
Modified: 1/2019

This file implements utility functions for Ozkan Lab interactome work.

Todo:
    * Test exhaustively (through main method)

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import scipy.io as sio
import math
import copy
from utilities import save_edgelist


################################################################################
def form_by_sparsity(W, pct, save_fn=None):
    """ Forms graph from weight matrix W given sparsity percentage pct.

        Args:    W       (numpy.ndarray)
                 pct     (float)
                 save_fn (None or str)
        
        Returns: graph   (numpy.ndarray)

    """ 
    
    # do some argument verification
    assert pct >= 0 and pct <= 1, "Argument `pct` should be in range [0, 1]."
    assert type(W) == np.ndarray, "Argument `adj` should be numpy.ndarray."

    # select top `pct` % of links
    threshold_weight = np.percentile(W.flatten(), 100 - 100 * pct)
    graph = copy.deepcopy(W)
    graph[graph < threshold_weight] = 0

    # save if desired
    if save_fn is not None:
        save_edgelist(graph, save_fn)

    return graph

################################################################################
def form_by_threshold(W, thresh, save_fn=None):
    """ Forms graph from weight matrix W given edge-weight threshold value.

        Args:    W       (numpy.ndarray)
                 thresh  (float)
                 save_fn (None or str)
        
        Returns: graph   (numpy.ndarray)

    """ 
    
    # do some argument verification
    assert type(W) == np.ndarray, "Argument `W` should be numpy.ndarray."
    assert thresh < np.amax(W), "Argument `thresh` should be < max(W)."

    # call any edge with weight > threshold a link
    graph = copy.deepcopy(W)
    graph[graph < thresh] = 0

    # save if desired
    if save_fn is not None:
        save_edgelist(graph, save_fn)

    return graph


