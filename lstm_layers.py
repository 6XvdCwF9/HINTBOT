#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.training import moving_averages
from layer import Layer

def uniform(shape, scale=0.05, name=None):
	"""Uniform init."""
	initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def glorot(shape, name=None):
	"""Glorot & Bengio (AISTATS 2010) init."""
	# init_range = np.sqrt(6.0/(shape[0]+shape[1]))
	init_range = np.sqrt(3.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	# initial = tf.random_normal(shape, mean=0.0, stddev=np.sqrt(2.0/(shape[0]+shape[1])), dtype=tf.float32)
	return tf.Variable(initial, name=name)

def zeros(shape, name=None):
	"""All zeros."""
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def ones(shape, name=None):
	"""All ones."""
	initial = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)
_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
	"""Helper function, assigns unique layer IDs."""
	if layer_name not in _LAYER_UIDS:
		_LAYER_UIDS[layer_name] = 1
		return 1
	else:
		_LAYER_UIDS[layer_name] += 1
		return _LAYER_UIDS[layer_name]

class LSTMLearnerLayer(Layer):

	def __init__(self, input_dim, num_time_steps, act=tf.sigmoid, name=None, **kwargs):
		super(LSTMLearnerLayer, self).__init__(**kwargs)
		allowed_kwargs = {'name', 'logging', 'model_size'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			layer = self.__class__.__name__.lower()
			name = layer + '_' + str(get_layer_uid(layer))
		self.name = name
		self.vars = {}
		logging = kwargs.get('logging', False)
		self.logging = logging
		self.input_dim = input_dim
		self.num_time_steps = num_time_steps
		self.act = act

		if name is not None:
			name = '/' + name
		else:
			name = ''
		input_dim1=64
		with tf.variable_scope(self.name + name + '_vars'):
			self.vars['lstm_weights_gx'] = glorot([input_dim, input_dim], name='lstm_weights_gx')
			self.vars['lstm_weights_gh'] = glorot([input_dim, input_dim], name='lstm_weights_gh')
			self.vars['lstm_weights_ix'] = glorot([input_dim, input_dim], name='lstm_weights_ix')
			self.vars['lstm_weights_ih'] = glorot([input_dim, input_dim], name='lstm_weights_ih')
			self.vars['lstm_weights_fx'] = glorot([input_dim, input_dim], name='lstm_weights_fx')
			self.vars['lstm_weights_fh'] = glorot([input_dim, input_dim], name='lstm_weights_fh')
			self.vars['lstm_weights_ox'] = glorot([input_dim, input_dim], name='lstm_weights_ox')
			self.vars['lstm_weights_oh'] = glorot([input_dim, input_dim], name='lstm_weights_oh')

			self.vars['lstm_bias_g'] = zeros([input_dim], name='lstm_bias_g')
			self.vars['lstm_bias_i'] = zeros([input_dim], name='lstm_bias_i')
			self.vars['lstm_bias_f'] = zeros([input_dim], name='lstm_bias_f')
			self.vars['lstm_bias_o'] = zeros([input_dim], name='lstm_bias_o')

		if self.logging:
			self._log_vars()

	def _call(self, inputss):
		# inputss: [sample,6,32] neg:[5,6,32]
		# print(inputss.shape)
		graphs = tf.transpose(inputss, perm=[1,2,0]) #(6,32,sample)
		# print(graphs.shape)
		state_h_t = tf.zeros_like(graphs[0,:,:]) #(32,sample)
		state_s_t = tf.zeros_like(graphs[0,:,:]) #(32,sample)
		outputs = []

		for idx in range(0, self.num_time_steps):

			inputs = graphs[idx,:,:] #(32,sample)

			gate_g_ = tf.matmul(self.vars['lstm_weights_gx'], inputs) #[32,sample]
			gate_g_ += tf.matmul(self.vars['lstm_weights_gh'], state_h_t) 
			gate_g_ = tf.transpose(gate_g_)+self.vars['lstm_bias_g']
			gate_g = tf.transpose(tf.nn.tanh(gate_g_))

			gate_i_ = tf.matmul(self.vars['lstm_weights_ix'], inputs) #[32,sample]
			gate_i_ += tf.matmul(self.vars['lstm_weights_ih'], state_h_t)
			gate_i_ = tf.transpose(gate_i_)+self.vars['lstm_bias_i']
			gate_i = tf.transpose(tf.sigmoid(gate_i_))

			gate_f_ = tf.matmul(self.vars['lstm_weights_fx'], inputs) #[32,sample]
			gate_f_ += tf.matmul(self.vars['lstm_weights_fh'], state_h_t) 
			gate_f_ = tf.transpose(gate_f_)+self.vars['lstm_bias_f']
			gate_f = tf.transpose(tf.sigmoid(gate_f_))

			gate_o_ = tf.matmul(self.vars['lstm_weights_ox'], inputs) #[32,sample]
			gate_o_ += tf.matmul(self.vars['lstm_weights_oh'], state_h_t)
			gate_o_ = tf.transpose(gate_o_)+self.vars['lstm_bias_o']
			gate_o = tf.transpose(tf.sigmoid(gate_o_))

			state_s = tf.multiply(gate_g, gate_i) + tf.multiply(state_s_t, gate_f)
			state_h = tf.multiply(state_s, gate_o)

			state_h_t = state_h
			state_s_t = state_s
			outputs.append(state_h)

		outputs = tf.transpose(outputs, perm=[2,0,1])

		return outputs

	def _log_vars(self):
		pass

