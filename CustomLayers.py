# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:45:53 2021
@author: Joe Raso
"""

import tensorflow as tf

class VariationalLatentLayer(tf.keras.layers.Layer):
    """ Custom variational latent layer, used for constructing variational
        autoencoders. Connects two equal-dimension layers (the log-variance
        and mean) and feeds them into a randomly sampled latent space using the
        'reparametrization trick':
        
            z = VariationalLatentLayer()([logvar, mu])
        
        Incorporating this layer into a model also automatically adds the
        appropriate KL divergence to the losses in the model.
        
        New Keywords (in addition to those from tf.keras.layers.Layer):
        track_kl - (bool) If True, also adds the KL divergence to the metrics
                   of the model the layer is incorporated into under the name
                   'kl_divergence'. (Default is True.)
        """
    def __init__(self, track_kl=True, **kwargs):
        super().__init__(**kwargs)
        self.track_kl = track_kl
    def call(self, inputs):
        # retrieve the inputs
        logvar, mu = inputs
        # generate a random sample from a normal distribution
        epsilon = tf.keras.backend.random_normal(tf.keras.backend.shape(mu))
        # calculate the variance
        sigma = tf.keras.backend.exp(logvar)
        # add KL divergence to the loss
        KLloss = -0.5 * tf.keras.backend.sum(1 + logvar - 
            tf.keras.backend.square(mu) - tf.keras.backend.exp(logvar), axis=1)
        self.add_loss(tf.keras.backend.mean(KLloss))
        # add kl divergence to metrics, if desired
        if self.track_kl:
            self.add_metric(tf.keras.backend.mean(KLloss),name='kl_divergence')
        # return the latent space
        return mu + sigma*epsilon            