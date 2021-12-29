import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class OffensiveCNNModel:
    def __init__(self, args, embedding_matrix=None):
        filter_sizes = [1, 2, 3, 5]
        num_filters = 32

        inp = tf.keras.Input(shape=(None,), dtype="int64", name="input")
        x = layers.Embedding(args.max_features, args.embed_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(inp)
        x = layers.SpatialDropout1D(0.4)(x)
        x = layers.Reshape((args.max_len, args.embed_size, 1))(x)

        conv_0 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_1 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_2 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_3 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[3], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)

        maxpool_0 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[0] + 1, 1))(conv_0)
        maxpool_1 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[1] + 1, 1))(conv_1)
        maxpool_2 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[2] + 1, 1))(conv_2)
        maxpool_3 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[3] + 1, 1))(conv_3)

        z = layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = layers.Flatten()(z)
        z = layers.Dropout(0.1)(z)
        outp = layers.Dense(args.num_classes, activation="softmax", name="dense_predictions")(z)
        self.model = tf.keras.Model(inputs=inp, outputs=outp, name="cnn_model")
