from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention


class OffensiveLSTMModel:
    def __init__(self, args, embedding_matrix):
        inp = Input(shape=(args.max_len,))
        x = Embedding(args.max_features, args.embed_size, weights=[embedding_matrix], trainable=False)(inp)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(64))(x)
        # x = Attention(args.max_len)(x)
        x = Dense(256, activation="relu")(x)
        # x = Dropout(0.25)(x)
        x = Dense(args.num_classes, activation="sigmoid")(x)
        self.model = Model(inputs=inp, outputs=x)
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
