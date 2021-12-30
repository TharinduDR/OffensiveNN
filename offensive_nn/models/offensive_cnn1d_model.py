import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class OffensiveCNN1DModel:
    def __init__(self, args, embedding_matrix=None):

        int_sequences_input = keras.Input(shape=(None,), dtype="int64")
        embedded_sequences = layers.Embedding(args.max_features, args.embed_size,
                             embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,
                             name="embedding_layer")(int_sequences_input)
        x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        preds = layers.Dense(args.num_classes, activation="softmax", name="dense_predictions")(x)
        self.model = keras.Model(int_sequences_input, preds, name="cnn1D_model")
