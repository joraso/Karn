# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:46:41 2021
@author: Joe Raso
"""

import numpy as np
import tensorflow as tf
from utils import connect

class SimpleVariationalAutoEncoder:
    """ A variational version of the simple autoencoder neural network design.
        The basic construction is:
        
            input -> encoder -> (mu/logvar) -> latent -> decoder -> output
                              
        
        The encoder consists of an input layer, followed by any number of
        interstitial dense hidden layers (specified by 'structure'), which are
        connected to a pair of then a low dimensional layers representing the
        mean (mu) and log-variance. The latent representation is sampled
        from a Gaussian with these params. The decoder is a series of dense
        layers (in mirror to the encoder) that reconstructs the original
        vector. The model is optimized with respect to both the reconstruction
        loss (binary cross-entropy between input and output) and the KL
        divergence between the latent space distribution and a normal prior.
        
        Arguments:
        inputdims   - (int) Dimensions of the input data. Note that this
                      network does not perform any data reshaping.
        latentdims  - (int) Dimension of the latent space. Each dimension
                      having it's own mu and logvar.
        blocks      - (list) List of DenseBlock building block objects to
                      construct the (encoding) network from.
                      
        Keywords:
        beta        - (float) Weight to give to the KL loss term. Should be
                      alterable throughout training. Defaults to 1.
        decode_blocks - (list) List of DenseBlock building block objects to
                      construct the decoding network from. If None (default),
                      the decoder is built as a mirror image of the encoder.
        optimizer   - (str) Keras optimizer to use during training. defaults
                      to 'rmsprop', which is apparently the optimal choice for
                      variational autoencoder models.
        latent_activation - (str) Activation function to use in the logvar and
                      mu layers. Defaults to 'relu'.
        output_activation - (str) Activation function to use in the output
                      layer. Defaults to 'sigmoid'.
        verbose     - (bool) Toggles command line output. Defaults to True.
        """
    def __init__(self, inputdims, latentdims, blocks, beta=1.0,
                 decode_blocks=None, optimizer='rmsprop',
                 latent_activation='relu', output_activation='sigmoid', 
                 verbose=True):
        # Note: input does not perform reshaping
        self.inputdims = inputdims
        self.latentdims = latentdims
        self.blocks = blocks
        self.beta = beta
        self.decode_blocks = decode_blocks
        self.optimizer = optimizer
        self.latent_activation = latent_activation
        self.output_activation = output_activation
        self.verbose = verbose
        # Attributes not created by the user
        self.layers = []; self.model = None
        # Loss tracking
        self.totalloss=[]; self.klloss=[]; self.reconloss=[]
        # Do the thing:
        self.Build()
        
    def Build(self):
        """Constructs the model from the specified structure."""
        # Define the encoder layers:
        self.input = tf.keras.Input(shape=self.inputdims) 
        self.encoding_layers = []
        for blk in self.blocks:
            self.encoding_layers += blk.layerlist()
        # The latent space now consists of sigma (as logvar) and mu.
        self.logvar = tf.keras.layers.Dense(self.latentdims,
                activation = self.latent_activation)
        self.mu = tf.keras.layers.Dense(self.latentdims,
                activation = self.latent_activation)
        
        # Build to encoder sigma (logvar) and mu outputs
        encode = connect(self.input, self.encoding_layers)
        self.logvar = self.logvar(encode)
        self.mu = self.mu(encode)
        
        # Finally, the variational latent layer
        def sampling(inputs):
            # retrieve the inputs
            logvar, mu = inputs
            # generate a random sample from a normal distribution
            epsilon = tf.keras.backend.random_normal(tf.keras.backend.shape(mu))
            # calculate the variance
            sigma = tf.keras.backend.exp(logvar)
            # Resturn a scaled z
            return mu + sigma*epsilon
        self.latent = tf.keras.layers.Lambda(sampling)([self.logvar, self.mu])
        
        # and build the encoder
        self.encoder = tf.keras.Model(self.input, self.latent)
        
        # Define the decoding layers:
        self.encoded_input = tf.keras.Input(shape=self.latentdims) 
        self.decoding_layers = []
        if self.decode_blocks != None:
            for blk in self.decode_blocks:
                self.decoding_layers += blk.layerlist()
        else:
            for blk in self.blocks:
                self.decoding_layers += blk.layerlist(reverse=True)        
        self.output = tf.keras.layers.Dense(self.inputdims,                                            
                activation = self.output_activation)
        self.decoding_layers.append(self.output)
        
        # Build the decoder
        decoded = connect(self.encoded_input, self.decoding_layers)
        self.decoder = tf.keras.Model(self.encoded_input, decoded)
        
        # Implement the KL loss funtion
        def divergence(inputs):
            # retrieve the inputs
            logvar, mu = inputs
            # Calculate the by-sample loss
            KLloss = -0.5 * self.beta * tf.keras.backend.sum(1 + logvar - 
            tf.keras.backend.square(mu) - 
            tf.keras.backend.exp(logvar), axis=1)
            # return the mean divergence
            return tf.keras.backend.mean(KLloss)
        self.KLloss = tf.keras.layers.Lambda(divergence)([self.logvar,self.mu])
        
        # Compile the auto encoder for training
        autoencoded = connect(self.latent, self.decoding_layers)
        self.autoencoder = tf.keras.Model(self.input, autoencoded)
        self.autoencoder.compile(optimizer=self.optimizer,
            loss='BinaryCrossentropy',
            metrics=[tf.keras.metrics.BinaryCrossentropy()])
        # add in the KLloss / Metric
        self.autoencoder.add_loss(self.KLloss)
        self.autoencoder.add_metric(self.KLloss, name='kl_divergence')

    def train(self, xtrain, batch=1, epochs=1):
        """ Trains the model on the given data set.
        
            Arguments:
            xtrain - (ndarray) Input training data.
            
            Keywords:
            batch  - (int) Batch size. defaults to 1.
            epochs - (int) Number of epochs. defaults to 1.
            """
        self.autoencoder.fit(xtrain, xtrain, batch_size=batch, epochs=epochs,
                       verbose=self.verbose, shuffle=True)
        self.totalloss += (self.autoencoder.history.history['loss'])
        self.klloss += (self.autoencoder.history.history['kl_divergence'])
        self.reconloss += (self.autoencoder.history.history['binary_crossentropy'])
    
    def Encode(self, xdata):
        """ Returns the latent space/encoded representation of the data.
        
            Arguments:
            xdata - (ndarray) Input data.
            
            Returns:
            xdata - (ndarray) Encoded representation of the data.
            """
        return self.encoder.predict(xdata)
    def Decode(self, zdata):
        """ Returns the reconstruction of output data based on space/encoded
            representation.
            
            Arguments:
            zdata - (ndarray) Input encoded data.
            
            Returns:
            xdata - (ndarray) Decoded/reconstructed output data.
            """
        return self.decoder.predict(zdata)