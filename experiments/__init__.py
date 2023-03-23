SEED = 42
THREADS = 3

import tensorflow as tf
import os

# From npfl114
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(THREADS)
tf.config.threading.set_intra_op_parallelism_threads(THREADS)
