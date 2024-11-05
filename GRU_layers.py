import numpy as np
import tensorflow as tf
from layer import Layer

def uniform(shape, scale=0.05, name=None):
	initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def glorot(shape, name=None):
	init_range = np.sqrt(3.0/(shape[0]+shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)
def zeros(shape, name=None):
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def ones(shape, name=None):
	initial = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)
_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
	if layer_name not in _LAYER_UIDS:
		_LAYER_UIDS[layer_name] = 1
		return 1
	else:
		_LAYER_UIDS[layer_name] += 1
		return _LAYER_UIDS[layer_name]


class GRULearnerLayer(Layer):
    def __init__(self, input_dim, num_time_steps, act=tf.sigmoid, name=None, **kwargs):
        super(GRULearnerLayer, self).__init__(**kwargs)
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
        input_dim1 = 64
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['gru_weights_zx'] = glorot([input_dim, input_dim], name='gru_weights_zx')
            self.vars['gru_weights_zh'] = glorot([input_dim, input_dim], name='gru_weights_zh')
            self.vars['gru_weights_rx'] = glorot([input_dim, input_dim], name='gru_weights_rx')
            self.vars['gru_weights_rh'] = glorot([input_dim, input_dim], name='gru_weights_rh')
            self.vars['gru_weights_hx'] = glorot([input_dim, input_dim], name='gru_weights_hx')
            self.vars['gru_weights_hh'] = glorot([input_dim, input_dim], name='gru_weights_hh')
            self.vars['gru_bias_z'] = zeros([input_dim], name='gru_bias_z')
            self.vars['gru_bias_r'] = zeros([input_dim], name='gru_bias_r')
            self.vars['gru_bias_h'] = zeros([input_dim], name='gru_bias_h')

        if self.logging:
            self._log_vars()

    def _call(self, inputss):
        graphs = tf.transpose(inputss, perm=[1, 2, 0])  # (6,32,sample)
        state_h_t = tf.zeros_like(graphs[0, :, :])  # (32,sample)
        outputs = []

        for idx in range(0, self.num_time_steps):
            inputs = graphs[idx, :, :]  # (32,sample)

            gate_z_ = tf.matmul(self.vars['gru_weights_zx'], inputs)  # [32,sample]
            gate_z_ += tf.matmul(self.vars['gru_weights_zh'], state_h_t)
            gate_z_ = tf.transpose(gate_z_) + self.vars['gru_bias_z']
            gate_z = tf.transpose(tf.sigmoid(gate_z_))

            gate_r_ = tf.matmul(self.vars['gru_weights_rx'], inputs)  # [32,sample]
            gate_r_ += tf.matmul(self.vars['gru_weights_rh'], state_h_t)
            gate_r_ = tf.transpose(gate_r_) + self.vars['gru_bias_r']
            gate_r = tf.transpose(tf.sigmoid(gate_r_))

            h_tilde_ = tf.matmul(self.vars['gru_weights_hx'], inputs)  # [32,sample]
            h_tilde_ += tf.matmul(self.vars['gru_weights_hh'], tf.multiply(gate_r, state_h_t))
            h_tilde_ = tf.transpose(h_tilde_) + self.vars['gru_bias_h']
            h_tilde = tf.transpose(tf.tanh(h_tilde_))
            state_h = tf.multiply(gate_z, state_h_t) + tf.multiply(1 - gate_z, h_tilde)
            state_h_t = state_h
            outputs.append(state_h)
        outputs = tf.transpose(outputs, perm=[2, 0, 1])
        return outputs
    def _log_vars(self):
        pass
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
		graphs = tf.transpose(inputss, perm=[1,2,0]) #(6,32,sample)
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

