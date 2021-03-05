# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:17 2021
@author: Joe Raso
"""

import tensorflow as tf
from utils import convdim

class DenseBlock:
    """ Object containing the parameters for a block of hidden dense layers.
        
        Arguments:
        structure    - (list) List of ints specifying the dimensionality 
                       (number of neurons) of each hidden layer in the block.
        
        Keywords:
        activation   - (str) Activation function to use in the hidden layers.
                       Defauls to 'relu'.
        dense_params - (dict) Dictionary of keyword arguments to pass to the
                       hidden layers (if different from keras defaults).
        """
    def __init__(self, structure, activation='relu', dense_params={}):
        self.structure = structure
        self.activation = activation
        self.dense_params = {'use_bias':True,
            'kernel_initializer':"glorot_uniform",
            'bias_initializer':"zeros",
            'kernel_regularizer':None, 'bias_regularizer':None,
            'activity_regularizer':None, 'kernel_constraint':None, 
            'bias_constraint':None}
        self.dense_params.update(dense_params)
    
    def layerlist(self, reverse=False):
        """ Consructs the keras layers comprising the block hidden layers and
            returns them as list to be compiled by the nework object.
            
            Keywords:
            reverse   - (bool) If True, returns a list of layers in the reverse
                        order. Defaults to False.
            """
        layers = []; structure = self.structure.copy()
        # processing the 'reverse' keyword
        if reverse:
            structure.reverse()
        # Building the list
        for i in structure:
            layers.append(tf.keras.layers.Dense(i,
                activation = self.activation, **self.dense_params))
        return layers
        
class Conv2Dblock:
    """ An object containing the parameters of a compound convolutional layer,
        consisting of a Conv2D layer + a 2D pooling layer.
        
        Arguments:
        filters     - (int) The dimensionality of the output space of the
                      convulutional piece of the block.
        kernel_size - (tuple) The kernal size of the convolution.
        
        Keywords:
        activation  - (str) Activation function to use in the convolutional
                      layers. Defauls to 'relu'.
        pooltype    - (str) The type of pooling to be performed in the pooling
                      layer. Current options are: 
                          'max' - Max pooling
                          'ave' - average pooling
                          'ups' - upsampling
                      Defaults to 'max'.
        inversion  - (str) Specify the way the block should be inverted.
                      options currently include:
                          'upsample' - convolution + upsampling
                          'deconv'   - deconvolution + normalizing
                      Defaults to upsampling.
        pool_size   - (tuple) The (vertical, horizontal) factors by which to
                      downscale in the pooling layer. Defaults to (2,2).
        conv_params - (dict) Dictionary of keyword parameters to pass to the
                      convolutional layer (if different from keras defaults).
        pool_params - (dict) Dictionary of keyword parameters to pass to the
                      average or max pooling layer (if different from keras
                      defaults).
        ups_params  - (dict) Dictionary of keyword parameters to pass to the
                      up-sampling layer in inverted layers (if different from
                      keras defaults).
        """
    def __init__(self, filters, kernel_size, activation='relu', pooltype='max',
                 inversion='upsample', pool_size=(2,2), conv_params={},
                 pool_params={}, ups_params={}):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.inversion = inversion
        self.pooltype = pooltype
        if self.pooltype not in ['max', 'ave', 'ups']:
            print("Error: unrecognized pooling type: {}.".format(self.pool_type))
        self.pool_size = pool_size
        self.conv_params = {'strides':(1, 1), 'padding':"valid",
            'data_format':'channels_last',
            'kernel_initializer':"glorot_uniform",
            'bias_initializer':"zeros",
            'dilation_rate':(1, 1), 'groups':1, 'use_bias':True,
            'kernel_regularizer':None, 'bias_regularizer':None,
            'activity_regularizer':None, 'kernel_constraint':None, 
            'bias_constraint':None}
        self.conv_params.update(conv_params)
        self.pool_params = {'strides':None, 'padding':"valid",
            'data_format':'channels_last'}
        self.pool_params.update(pool_params)
        self.ups_params = {'interpolation':"nearest",
            'data_format':'channels_last'}
        self.ups_params.update(ups_params)
        
    def inverse(self):
        """ Returns a BuildingBlock object that is the inverse according to the
            selected inversion type.
            
            Returns:
            inverse - (object) Building block for the inverse.
            """
        if self.inversion == 'upsample':
            return Conv2Dblock(self.filters, self.kernel_size, 
                        activation=self.activation, pooltype='ups',
                        pool_size=self.pool_size, conv_params=self.conv_params,
                        pool_params=self.pool_params,
                        ups_params=self.ups_params)
        elif self.inversion == 'deconv':
            return Deconv2Dblock(self.filters, self.kernel_size,
                        self.pool_size, activation=self.activation,
                        conv_params={})
        else: # toss error
            print("Error: Inversion type not recognized.")

    def layerlist(self):
        """ Constructs the keras layers comprising the convolutional block and
            returns a (2-member) list to be compiled by the nework object.
            
            Keywords:
            inverted - (bool) If True, returns an iverted convolution (up-
                       sampling followed by deconvolution). Defaults to False.
                       
            Returns:
            layers   - (list) A list of un-built keras layers.
            """
        convlayer = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
            activation = self.activation, **self.conv_params)
        if self.pooltype == 'max':
            pool = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size,
                **self.pool_params)
        elif self.pooltype == 'ave':
            pool = tf.keras.layers.AveragePooling2D(
                pool_size=self.pool_size, **self.pool_params)
        elif self.pooltype == 'ups':
            pool = tf.keras.layers.UpSampling2D(size=self.pool_size,
                **self.ups_params)
        return [convlayer, pool]
        
    def output_shape(self, input_shape):
        """ Attempts to predict the output tensor shape of the block. (not
            including the channel dimension.) Although untested, this should 
            now account for different padding types and pool strides (but
            not dilation rate, yet).
        
            Arguments:
            input_shape  - (tuple, len=3) The input shape anitcipated from the
                           previous layer/block.
        
            Returns:
            output_shape - (tuple, len=3) The predicted output dimensions of
                           the convolution block.
            """
        # Accounting for convolution
        x = convdim(input_shape[0], self.kernel_size[0],
            self.conv_params['strides'][0],
            padding=self.conv_params['padding'])
        y = convdim(input_shape[1], self.kernel_size[1],
            self.conv_params['strides'][1],
            padding=self.conv_params['padding'])
        # Accounting for pooling
        if self.pooltype == 'ups':
            x = x*self.pool_size[0]; y = y*self.pool_size[1]
        else:
            if self.pool_params['strides'] == None:
                poolstride = self.pool_size
            else:
                poolstride = self.pool_params['strides']
            x = convdim(x, self.pool_size[0], poolstride[0],
                padding=self.pool_params['padding'])
            y = convdim(y, self.pool_size[1], poolstride[1],
                padding=self.pool_params['padding'])
        return (x, y, self.filters)
    
    def input_shape(self, output_shape):
        """ Attempts to predict the input tensor shape of the block for a given
            output shape. (The channel dimention is unchanged.) Although
            untested, this should now account for different padding types and
            pool strides (but not dilation rate, yet).
            
            Arguments:
            output_shape - (tuple, len=3) The target output shape.
            
            Returns:
            input_shape  - (tuple, len=3) The neccessary input dimensions to
                           produce the target output.
            """
        # Accounting for convolution
        x = convdim(output_shape[0], self.kernel_size[0],
            self.conv_params['strides'][0],
            padding=self.conv_params['padding'], reverse=True)
        y = convdim(output_shape[1], self.kernel_size[1],
            self.conv_params['strides'][1],
            padding=self.conv_params['padding'], reverse=True)
        # Accounting for pooling
        if self.pooltype == 'ups':
            x = int(x/self.pool_size[0]); y = int(y/self.pool_size[1])
        else:
            if self.pool_params['strides'] == None:
                poolstride = self.pool_size
            else:
                poolstride = self.pool_params['strides']
            x = convdim(x, self.pool_size[0], poolstride[0],
                padding=self.pool_params['padding'], reverse=True)
            y = convdim(y, self.pool_size[1], poolstride[1],
                padding=self.pool_params['padding'], reverse=True)
        return (x, y, self.filters)
        
class Deconv2Dblock:
    """ An object containing the parameters of a compound deconvolution layer,
        consisting of a Conv2DTranspose layer + a BatchNormalization layer.
        
        Arguments:
        filters     - (int) The dimensionality of the output space of the
                      convulutional piece of the block.
        kernel_size - (tuple) The kernal size of the convolution.
        stride      - (tuple) The stride of the deconvolution
        
        Keywords:
        activation  - (str) Activation function to use in the convolutional
                      layers. Defauls to 'relu'.
        conv_params - (dict) Dictionary of keyword parameters to pass to the
                      convolutional layer (if different from keras defaults).
        """
    def __init__(self, filters, kernel_size, stride, activation='relu',
                 conv_params={}):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.conv_params = {'padding':"valid",
            'data_format':'channels_last',
            'kernel_initializer':"glorot_uniform",
            'bias_initializer':"zeros",
            'dilation_rate':(1, 1), 'groups':1, 'use_bias':True,
            'kernel_regularizer':None, 'bias_regularizer':None,
            'activity_regularizer':None, 'kernel_constraint':None, 
            'bias_constraint':None}
        self.conv_params.update(conv_params)

    def layerlist(self, inverted=False):
        """ Constructs the keras layers comprising the convolutional block and
            returns a (2-member) list to be compiled by the nework object.
            
            Keywords:
            inverted - (bool) If True, returns an iverted convolution (up-
                       sampling followed by deconvolution). Defaults to False.
                       
            Returns:
            layers   - (list) A list of un-built keras layers.
            """
        
        deconvlayer = tf.keras.layers.Conv2DTranspose(self.filters,
            self.kernel_size, strides=self.stride, 
            activation = self.activation, **self.conv_params)
        norm = tf.keras.layers.BatchNormalization()
        return [deconvlayer, norm]
        
    def output_shape(self, input_shape, inverted=False):
        """ Attempts to predict the output tensor shape of the block. (not
            including the channel dimension.) Although untested, this should
            now account for different padding types (but not dilation rate,
            yet).
        
            Arguments:
            input_shape  - (tuple, len=3) The input shape anitcipated from the
                           previous layer/block.
        
            Returns:
            output_shape - (tuple, len=3) The predicted output dimensions of
                           the convolution block.
            """
        x = convdim(input_shape[0], self.kernel_size[0], self.stride[0],
            padding=self.conv_params['padding'], reverse=True)
        y = convdim(input_shape[1], self.kernel_size[1], self.stride[1],
            padding=self.conv_params['padding'], reverse=True)
        return (x, y, self.filters)
    
    def input_shape(self, output_shape, inverted=False):
        """ Attempts to predict the input tensor shape of the block for a given
            output shape. (The channel dimention is unchanged.) Although
            untested, this should now account for different padding types (but
            not dilation rate, yet).
            
            Arguments:
            output_shape - (tuple, len=3) The target output shape.
            
            Returns:
            input_shape  - (tuple, len=3) The neccessary input dimensions to
                           produce the target output.
            """
        x = convdim(output_shape[0], self.kernel_size[0], self.stride[0],
            padding=self.conv_params['padding'])
        y = convdim(output_shape[1], self.kernel_size[1], self.stride[1],
            padding=self.conv_params['padding'])
        return (x, y, self.filters)