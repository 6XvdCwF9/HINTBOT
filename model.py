import tensorflow as tf
from keras.layers import *
from tensorflow.contrib import rnn
from lstm_layers import GRULearnerLayer
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = inputs[0],inputs[1]
    position_j = 1. / tf.pow(10000.,
                             2 * tf.range(position_size / 2, dtype=tf.float32
                            ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding

def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

def Dense(inputs, ouput_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs,
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs

def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    Q = Dense(Q, nb_head * size_per_head, True)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, nb_head * size_per_head, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, nb_head * size_per_head, False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class HINTBOT(object):
    def __init__(self, config, sess, node_embed, **kwargs):
        self.n_sequences = config.n_sequences
        concat = True
        self.vars = {}
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.sequence_batch_size = config.sequence_batch_size
        self.batch_size = config.batch_size
        self.display_step = config.display_step
        self.embedding_size = config.embedding_size
        self.n_input = config.n_input
        self.n_steps = config.n_steps
        self.n_hidden_gru = config.n_hidden_gru
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.dropout_prob = config.dropout_prob
        self.sess = sess
        self.node_vec = node_embed
        self.name = "hintbot"
        self.build_input()
        self.build_var()
        self.pred = self.build_model()
        truth = self.y
        cost = tf.reduce_mean(tf.pow(self.pred - truth, 2)) + self.scale * tf.add_n(
            [self.regularizer(var) for var in tf.trainable_variables()])
        error = tf.reduce_mean(tf.pow(self.pred - truth, 2))
        mape=tf.reduce_mean(tf.abs((self.pred - truth))/truth)
        var_list1 = [var for var in tf.trainable_variables() if not 'embedding' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'embedding' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        grads = tf.gradients(cost, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)
        self.cost = cost
        self.error = error
        self.mape=mape
        self.train_op = train_op
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
    def build_input(self):
        self.x = tf.placeholder(tf.int32, [None, self.n_sequences, self.n_steps], name="x")
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        self.sz = tf.placeholder(tf.float32, [None, 1], name="sz")

    def build_var(self):
        with tf.variable_scope(self.name) as scope:
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', initializer=tf.constant(self.node_vec, dtype=tf.float32))
            with tf.variable_scope('BiGRU'):
                self.gru_fw_cell = rnn.GRUCell(self.n_hidden_gru)
                self.gru_bw_cell = rnn.GRUCell(self.n_hidden_gru)

            with tf.variable_scope('attention'):
                self.p_step = tf.get_variable('p_step', initializer=self.initializer([1, self.n_steps]),
                                              dtype=tf.float32)
                self.a_geo = tf.get_variable('a_geo', initializer=self.initializer([1]))
            with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.n_hidden_dense1])),
                    'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                             self.n_hidden_dense2])),
                    'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                    'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
                }

    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('hintbot') as scope:
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x),
                                             self.dropout_prob)

                with tf.variable_scope('BiGRU'):
                    x_vector = tf.transpose(x_vector, [1, 0, 2, 3])

                    x_vector = tf.reshape(x_vector, [-1, self.n_steps, self.n_input])

                    x_vector = tf.transpose(x_vector, [1, 0, 2])

                    x_vector = tf.reshape(x_vector, [-1, self.n_input])

                    x_vector = tf.split(x_vector, self.n_steps,0)

                    outputs, _, _ = rnn.static_bidirectional_rnn(self.gru_fw_cell, self.gru_bw_cell, x_vector,
                                                          dtype=tf.float32)

                    hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])

                    hidden_states = tf.transpose(
                        tf.reshape(hidden_states, [self.n_sequences, -1, self.n_steps, 2 * self.n_hidden_gru]),
                        [1, 0, 2, 3])
                with tf.variable_scope('GRU'):
                    dim_mult = 2
                    hidden_states = tf.reshape(hidden_states, (-1, 14, 10))
                    self.outputs1_temporal, self.gru_learner = \
                        self.gru_propogation(inputs=hidden_states, input_dim=10, num_time_steps=7)
                    state1 = self.outputs1_temporal
                    self.rnn_learner = self.gru_learner
                with tf.variable_scope('cell'):
                    self.add_cell()
                with tf.variable_scope('mmsa'):
                    print(state1.shape)

                    hidden_states=tf.reshape(state1,(16,5,2,14))
                    O_seq = Attention(hidden_states, hidden_states, hidden_states,8,16)
                    O_seq = GlobalAveragePooling1D()(O_seq)
                    O_seq = Dropout(0.5)(O_seq)
                    O_seq=tf.reshape(O_seq, (-1,64))

                with tf.variable_scope('dense'):
                    dense1 = self.activation(
                        tf.add(tf.matmul(O_seq, self.weights['dense1']), self.biases['dense1']))
                    dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
                    pred = self.activation(tf.add(tf.matmul(dense2, self.weights['out']), self.biases['out']))

                return pred

    cell_size=10
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.y = tf.expand_dims(self.y, axis=2)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.y, initial_state=self.cell_init_state, time_major=False)

    def train_batch(self, x, y, sz):
        y = np.reshape(y, (-1,1,1))
        x=np.reshape(x,(2,7,5))
        self.sess.run(self.train_op, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_error(self, x, y, sz):
        y = np.reshape(y, (-1,1,1))
        x = np.reshape(x, (2,7,5))
        return self.sess.run(self.error, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_mape(self, x, y, sz):
        y = np.reshape(y, (-1,1,1))
        x = np.reshape(x, (2,7,5))
        return self.sess.run(self.mape, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def gru_propogation(self, inputs, input_dim, num_time_steps, gru_learners=None, name=None):
        new_agg = gru_learners is None
        if new_agg:
            gru_learners = []
            learner = GRULearnerLayer(input_dim=input_dim, num_time_steps=num_time_steps)
            gru_learners.append(learner)
        gru_inputs = inputs
        for learner in gru_learners:
            gru_outputs = learner(gru_inputs)
            gru_inputs = gru_outputs
        return gru_outputs, gru_learners

