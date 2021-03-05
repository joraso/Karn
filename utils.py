# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:42:26 2021
@author: Joe Raso
"""

def connect(input_layer, layers):
    """ A simple tool for linking together keras layers.
        
        Arguments:
        input_layer - (keras input layer / tf tensor) The input layer for the
                      the stack.
        layers      - (list of uninitialized keras layers) The layers to be 
                      linked togerther, in order.
                      
        Returns:
        layer       - (initialized keras layer / tf tensor) the output 
                      resulting output layer."""
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer
    
def convdim(i, k, s, padding='valid', reverse=False):
    """ Calculates the 1D ouput dimension of a convolusion/deconvolution. The
        supporting info for this arithmetic can be found here:
        https://arxiv.org/pdf/1603.07285v1.pdf
        
        Arduments:
        i       - (int, >0) Input dimension.
        k       - (int, >0) Kernel along the corrosponding dimension.
        s       - (int, >0) Stride along the corrosponding dimension.
        
        Keywords:
        padding - (str) Keras padding method. Options are:
                    'valid' - (Default) No padding
                    'same'  - half/padding (same dims for odd k)
        reverse - (bool) If true, return the deconvolution dimension instead.
                  (Default is False.)
                  
        Returns:
        o       - (int, >0) The output dimension.
        """
    # First, establish the padding parameter:        
    if padding == 'same':
        p = int(k/2) # padding needed to produce the same dims (for odd k).
    else: # default to 'valid for unrecognized padding mode.
        p = 0 # 'valid' = no padding
    # Now the calculation
    if reverse: # Return the deconvolution dimensions
        return s*(i-1) + k - 2*p
    else: # Return the convolution dimensions
        return int((i+2*p-k)/s) + 1    