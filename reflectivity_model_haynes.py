import sys
import os
import numpy as np
np.random.seed(42)

import tensorflow as tf
from tensorflow import keras
from bumps.initpop import lhs_init
import pandas as pd

import json
import refl1d
from refl1d.names import *

from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

def calculate_reflectivity(q, model_description, q_resolution=0.02, sld=True):
    """
        Reflectivity calculation using refl1d
    """
    zeros = np.zeros(len(q))
    dq = q_resolution * q / 2.355

    # The QProbe object represents the beam
    probe = QProbe(q, dq, data=(zeros, zeros))
    #probe.oversample(11, seed=1)

    layers = model_description['layers']
    sample = Slab(material=SLD(name=layers[0]['name'],
                               rho=layers[0]['sld']), interface=layers[0]['roughness'])
    # Add each layer
    for l in layers[1:]:
        sample = sample | Slab(material=SLD(name=l['name'],
                               rho=l['sld'], irho=l['isld']),
                               thickness=l['thickness'], interface=l['roughness'])

    probe.background = Parameter(value=model_description['background'], name='background')
    expt = Experiment(probe=probe, sample=sample)

    q, r = expt.reflectivity()

    if sld:
        z, sld, _ = expt.smooth_profile()
    else:
        z = sld = None

    return model_description['scale'] * r, z, sld


class ReflectivityModels(object):
    model_description = dict(layers=[
                                dict(sld=2.07, isld=0, thickness=0, roughness=11.1, name='substrate'),
                                dict(sld=7.53, isld=0, thickness=162.4, roughness=21.9, name='bulk'),
                                dict(sld=4.79, isld=0, thickness=200.3, roughness=24.9, name='oxide'),
                                dict(sld=0, isld=0, thickness=0, roughness=0, name='air')
                         ],
                         scale=1,
                         background=0,
                        )
    parameters = [
                  dict(i=0, par='roughness', bounds=[0, 40]),
                  dict(i=1, par='sld', bounds=[0, 10]),
                  dict(i=1, par='thickness', bounds=[20, 300]),
                  dict(i=1, par='roughness', bounds=[0, 40]),
                  dict(i=2, par='sld', bounds=[1, 10]),
                  dict(i=2, par='thickness', bounds=[50, 300]),
                  dict(i=2, par='roughness', bounds=[0, 40]),
                 ]

    _config_name = 'haynes'

    def __init__(self, q=None, all_data=False, name='haynes'):
        self.all_data = all_data
        self._pars_array = []
        self._refl_array = []
        self._z_array = []
        self._sld_array = []

        self._train_pars = []
        self._train_data = None
        self._config_name = name

        if q is None:
            self.q = np.logspace(np.log10(0.005), np.log10(0.2), num=250)
        else:
            self.q = q

    def generate(self, n=100):
        """
            Generate a random sample of models
        """
        npars = len(self.parameters)

        # Generate values we will train with
        #self._train_pars = lhs_init(n, np.zeros(npars), [-np.ones(npars),np.ones(npars)], use_point=False)

        #t = tf.random.uniform(shape=[n, npars], minval=npars*[-1], maxval=npars*[1], seed=42, dtype=tf.dtypes.float16)
        #self._train_pars = tf.make_ndarray(tf.make_tensor_proto(t))

        self._train_pars = np.random.uniform(low=-1, high=1, size=[n, npars])

        # Compute model parameters and reflectivity using these values
        self.compute_reflectivity()

    def to_model_parameters(self, pars):
        """
            Transform an array of parameters to a list of calculable models
        """
        pars_array = np.zeros(pars.shape)

        for i, par in enumerate(self.parameters):
            a = (par['bounds'][1]-par['bounds'][0])/2.
            b = (par['bounds'][1]+par['bounds'][0])/2.
            pars_array.T[i] = pars.T[i] * a + b

        return pars_array

    def compute_reflectivity(self):
        """
            Transform an array of parameters to a list of calculable models
            and compute reflectivity
        """
        print("Computing reflectivity")
        self._pars_array = self.to_model_parameters(self._train_pars)

        # Compute reflectivity
        for p in self._pars_array:
            _desc = self.get_model_description(p)
            r, z, sld = calculate_reflectivity(self.q, _desc, sld=self.all_data)
            self._refl_array.append(r)
            if self.all_data:
                self._z_array.append(z)
                self._sld_array.append(sld)

    def get_model_description(self, pars):
        for i, par in enumerate(self.parameters):
            self.model_description['layers'][par['i']][par['par']] = pars[i]
        return self.model_description

    def get_preprocessed_data(self, errors=None):
        """
            Pre-process data
            If errors is provided, a random error will be added, taking the errors array
            as a relative uncertainty.
        """
        if errors is None:
            self._train_data = np.log10(self._refl_array*self.q**2/self.q[0]**2)
            #self._train_data = np.log10(self._refl_array*self.q**4/self.q[0]**4)
            #self._train_data = -np.log10(self._refl_array*self.q**4)
        else:
            _data = self._refl_array * (1.0 + np.random.normal(size=len(errors)) * errors)
            # Catch the few cases where we generated a negative intensity and take
            # the absolute value
            _data[_data<0] = np.fabs(_data[_data<0])
            self._train_data = np.log10(_data*self.q**2/self.q[0]**2)

        return self._train_pars, self._train_data

    def add_noise(self, error):
        """ Add noise according to the provided uncertainty array,
            which is the same length as the Q array
        """
        _data = np.asarray(self._refl_array) + np.random.normal(size=len(error)) * error
        return np.log10(_data*self.q**2/self.q[0]**2)

    def save(self, output_dir=''):
        """
            Save all data relevant to a training set
            @param output_dir: directory used to store training sets
        """
        # Save q values
        np.save(os.path.join(output_dir, "%s_q_values" % self._config_name), self.q)

        # Save training set
        if self._train_data is not None:
            np.save(os.path.join(output_dir, "%s_data" % self._config_name), self._train_data)
            np.save(os.path.join(output_dir, "%s_pars" % self._config_name), self._train_pars)

    def load(self, data_dir=''):
        self.q = np.load(os.path.join(data_dir, "%s_q_values.npy" % self._config_name))
        self._train_data = np.load(os.path.join(data_dir, "%s_data.npy" % self._config_name))
        self._train_pars = np.load(os.path.join(data_dir, "%s_pars.npy" % self._config_name))
        return self.q, self._train_data, self._train_pars


def save_model(model, model_name, data_dir=''):
    """
        Save a trained model
        @param model: TensorFlow model
        @param model_name: base name for saved files
        @param data_dir: output directory
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(data_dir, "%s.json" % model_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(data_dir, "%s.h5" % model_name))


def load_model(model_name, data_dir=''):
    """
        Load and return a trained model
        @param model_name: base name for saved files
        @param data_dir: directory containing trained model
    """
    # load json and create model
    json_file = open(os.path.join(data_dir, '%s.json' % model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(os.path.join(data_dir, '%s.h5' % model_name))
    return model
