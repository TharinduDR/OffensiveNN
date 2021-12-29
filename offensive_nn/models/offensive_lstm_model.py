import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class OffensiveLSTMModel:

    def __init__(self, args, embedding_matrix):
        inp = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(args.max_features, args.embed_size, embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False)(inp)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(args.num_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=x)
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
