# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:42:26 2021
@author: Joe Raso
"""

def connect(input_layer, layers):
    """A simple tool for linking together keras layers.
    
    Arguments:
    input_layer - (keras input layer / tf tensor) The input layer for the
                  the stack.
    layers      - (list of uninitialized keras layers) The layers to be linked
                  togerther, in order.
                  
    Returns:
    layer       - (initialized keras layer / tf tensor) the output resulting
                  output layer."""
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer