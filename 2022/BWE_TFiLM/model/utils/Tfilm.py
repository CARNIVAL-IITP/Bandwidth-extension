import tensorflow as tf

from tensorflow import keras
from keras import Model
from keras import layers
from keras.layers import LSTM, MaxPooling1D, Conv1D, \
    Input, LeakyReLU, Dropout, ReLU, Concatenate


def make_normalizer(x_in, n_filters, n_block):
    """applies an lstm layer on top of x_in"""
    # input (-1, 4096, n_filters) to (-1, 32, n_filters)
    # output: (-1, 32, n_filters)
    x_in_down = MaxPooling1D(n_block)(x_in)
    x_rnn = LSTM(units=n_filters, return_sequences=True)(x_in_down)
    return x_rnn


def apply_normalizer(x_in, x_norm, n_filters, n_block):
    x_shape = tf.shape(x_in)
    n_steps = x_shape[1] / n_block  # will be 32 at training

    # reshape input into blocks
    x_in = tf.reshape(x_in, shape=(-1, n_steps, n_block, n_filters))
    x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))

    # multiply
    x_out = x_norm * x_in

    # return to original shape
    x_out = tf.reshape(x_out, shape=x_shape)

    return x_out


def tfilm(x, filters, n_block):
    x_norm = make_normalizer(x, filters, n_block)
    x = apply_normalizer(x, x_norm, filters, n_block)
    return x


def SubPixel1D(I, r):
    with tf.name_scope('subpixel'):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.compat.v1.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X


def tfilm_net(filters=[128, 256, 512, 512, 512],
         sizes=[65, 33, 17, 9, 9]):
    down = []
    num_layers = len(filters) - 1
    assert len(filters) == len(sizes)

    # ********** input ********** #
    x_in = Input(shape=(8192, 1), name="input")
    x = x_in

    # ********** down-sampling layers ********** #
    for i in range(num_layers):
        x = Conv1D(filters=filters[i], kernel_size=sizes[i], padding="same",
                   dilation_rate=2, kernel_initializer="orthogonal")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = LeakyReLU(0.2)(x)
        nb = 128 // (2 ** i)
        x = tfilm(x, filters[i], nb)
        down.append(x)

    # ********** bottleneck layer ********** #
    x = Conv1D(filters=filters[-1], kernel_size=sizes[-1], padding="same",
               dilation_rate=2, kernel_initializer="orthogonal")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU(0.2)(x)
    nb = 128 // (2 ** (num_layers-1))
    x = tfilm(x, filters[-1], nb)

    # ********** up-sampling layers ********** #
    for i in range(num_layers):
        x = Conv1D(filters=2*filters[num_layers-i-1], kernel_size=sizes[num_layers-i-1], padding="same",
                   dilation_rate=2, kernel_initializer="orthogonal")(x)
        x = Dropout(0.5)(x)
        x = ReLU()(x)
        x = SubPixel1D(x, r=2)
        nb = 128 // (2 ** (num_layers-i-1))
        x = tfilm(x, filters[num_layers-i-1], nb)
        x = Concatenate(-1)([x, down[num_layers-i-1]])

    # ********** output ********** #
    x = Conv1D(filters=2, kernel_size=9, padding="same")(x)
    x = SubPixel1D(x, r=2)
    x_out = x + x_in

    return Model(x_in, x_out)


if __name__ == '__main__':
    model = tfilm_net()
    model.summary()