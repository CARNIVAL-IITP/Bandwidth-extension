import tensorflow as tf
import os, h5py, datetime

from time import time

from dataset import get_audio_dataset
from model.Tfilm import tfilm_net
from os import makedirs

from model.utils.losses import stoi_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_dataset_path = "./input/trainset.h5"
test_dataset_path = "./input/testset.h5"

model_name = "/logs/model.h5"

save_path = os.path.abspath(os.getcwd())

batch_size = 32
EPOCHS = 100
lr = 3e-4
alpha = 1e-1
SHUFFLE = True
CACHE = True

print(model_name)

strategy = tf.distribute.get_strategy()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/tensorboard/_' + current_time + model_name
makedirs(train_log_dir, exist_ok = True)

train_summary_writer = tf.summary.create_file_writer(train_log_dir)


if __name__ == '__main__':

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hf = h5py.File(train_dataset_path, 'r')
    length = hf['data'].shape[0]
    hf.close()

    with strategy.scope():
        if SHUFFLE:
            train_ds = get_audio_dataset(train_dataset_path, batch_size=batch_size, length=length, cache=CACHE)
        else:
            train_ds = get_audio_dataset(train_dataset_path, batch_size=batch_size, length=None, cache=CACHE)
        test_ds = get_audio_dataset(test_dataset_path, batch_size=batch_size, length=None, cache=CACHE)
    
    model = tfilm_net()

    loss_object = tf.keras.losses.MeanSquaredError() # MSE
    # loss_object = tf.keras.losses.MeanAbsoluteError() # MAE
    # loss_alpha = stoi_loss(batch_size, 63) # STOI loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    with strategy.scope():
        @tf.function
        def train_step(inpt, tagt):
            with tf.GradientTape() as tape:
                pred = model(inpt)
                loss = loss_object(tagt, pred)
                # loss = loss_object(tagt, pred) + alpha * loss_alpha(tagt, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

    with strategy.scope():
        @tf.function
        def test_step(inpt, tagt):
            pred = model(inpt)
           
            t_loss = loss_object(tagt, pred)
            # loss_alpha = stoi_loss(27,63)
            # t_loss = loss_object(tagt, pred) + alpha * loss_alpha(tagt, pred)
            test_loss(t_loss)

    best_accuracy = 1

    for epoch in range(EPOCHS):
        start = time()
        train_loss.reset_states()
        test_loss.reset_states()

        with strategy.scope():
            for inpt, tagt in train_ds:
                train_step(inpt, tagt)
                
            for inpt, tagt in test_ds:
                test_step(inpt, tagt)

        end = time()
        print('time elapsed:', str(datetime.timedelta(seconds = end - start)))
        a = test_loss.result()
        if a < best_accuracy:
            best_loss = test_loss.result()

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)

        model.save_weights(save_path + model_name)
        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              test_loss.result()))