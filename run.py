from alfred.dl.tf.common import mute_tf
mute_tf()
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from model import HINTBOT
import six.moves.cPickle as pickle
tf.set_random_seed(0)
import time
import math

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
DATA_PATH = ''
tf.flags.DEFINE_float("learning_rate", 0.1, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", 5, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size",7, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_integer("n_hidden_lstm", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-8, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("n_sequences", 7, "num of sequences.")
tf.flags.DEFINE_integer("training_iters", 200000, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("embedding_size", 128, "embedding size.")
tf.flags.DEFINE_integer("n_input", 50, "input size.")
tf.flags.DEFINE_integer("n_steps", 5, "num of step.")
tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", 5e-05, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", 0.2, "dropout probability.")

config = tf.flags.FLAGS

print('--------------------')
def get_batch(x, y, sz, step, batch_size=128):
    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_y = np.zeros((batch_size, 1))
    batch_sz = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)
    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_sz[i, 0] = sz[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])
    return batch_x, batch_y, batch_sz


version = config.version
x_train, y_train, sz_train, vocabulary_size = pickle.load(open(DATA_PATH + 'datatest/data_train.pkl', 'rb'))
x_test, y_test, sz_test, _ = pickle.load(open(DATA_PATH + 'datatest/data_test.pkl', 'rb'))
x_val, y_val, sz_val, _ = pickle.load(open(DATA_PATH + 'datatest/data_val.pkl', 'rb'))
node_vec = pickle.load(open(DATA_PATH + 'datatest/node_vec.pkl', 'rb'))
training_iters = config.training_iters
batch_size = config.batch_size
display_step=2
np.set_printoptions(precision=2)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = HINTBOT(config, sess, node_vec)
step = 1
best_val_loss = 1000
best_test_loss = 1000
best_val_mape=1000
best_test_mape=1000
best_test_r=1000
train_loss = []
train_mape=[]
train_r=[]
max_try = 138
patience = max_try
def trmape(a):
    for i in range(len(a)):
        a[i]=abs(a[i])
    return a

while step * batch_size < training_iters:

    batch_x, batch_y, batch_sz = get_batch(x_train, y_train, sz_train, step, batch_size=batch_size)
    model.train_batch(batch_x, batch_y, batch_sz)
    train_loss.append(model.get_error(batch_x, batch_y, batch_sz))
    train_mape.append(model.get_mape(batch_x, batch_y, batch_sz))

    if step % display_step == 0.0:
        val_loss = []
        val_mape=[]

        for val_step in range(math.ceil(len(y_val) / batch_size)):
            val_x, val_y, val_sz = get_batch(x_val, y_val, sz_val, val_step, batch_size=batch_size)
            val_loss.append(model.get_error(val_x, val_y, val_sz))
            val_mape.append(model.get_mape(val_x, val_y, val_sz))
        test_loss = []
        test_mape=[]


        for test_step in range(math.ceil(len(y_test) / batch_size)):
            test_x, test_y, test_sz = get_batch(x_test, y_test, sz_test, test_step, batch_size=batch_size)
            test_loss.append(model.get_error(test_x, test_y, test_sz))
            test_mape.append((model.get_mape(test_x, test_y, test_sz)))
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            best_test_mape=np.mean(trmape(test_mape))
            best_val_mape=np.mean(trmape(val_mape))
            patience = max_try

        print("#" + str(step / display_step) +
              ", Train_Loss= " + "{:.6f}".format(np.mean(train_loss)) +
              ", val_Loss= " + "{:.6f}".format(np.mean(val_loss)) +
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
              ", train_mape= " + "{:.6%}".format(np.mean(trmape(train_mape)))+
              ", val_mape= " + "{:.6%}".format(np.mean(trmape(val_mape)))+
              ", test_mape= " + "{:.6%}".format(np.mean(trmape(test_mape)))
        )

        train_loss = []

        patience -= 1
        if not patience:
            break
    step += 1
print("Finished!\n----------------------------------------------------------------")

