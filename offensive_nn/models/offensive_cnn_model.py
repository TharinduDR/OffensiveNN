import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class OffensiveCNNModel:
    def __init__(self, args, embedding_matrix=None):
        inp = keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(args.max_features, args.embed_size,
                         embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                         name="embedding_layer")(inp)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        preds = layers.Dense(args.num_classes, activation="softmax", name="dense_predictions")(x)
        self.model = tf.keras.Model(inputs=inp, outputs=preds, name="cnn_model")
