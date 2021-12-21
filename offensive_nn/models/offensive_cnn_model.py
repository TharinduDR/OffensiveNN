from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, SpatialDropout1D, Reshape, Conv2D, MaxPool2D, Concatenate, \
    Flatten, Dropout, Dense


class OffensiveCNNModel:

    def __init__(self, args, embedding_matrix):
        filter_sizes = [1, 2, 3, 5]
        num_filters = 32

        inp = Input(shape=(args.maxlen,))
        x = Embedding(args.max_features, args.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.4)(x)
        x = Reshape((args.maxlen, args.embed_size, 1))(x)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)
        conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], args.embed_size), kernel_initializer='normal',
                        activation='elu')(x)

        maxpool_0 = MaxPool2D(pool_size=(args.maxlen - filter_sizes[0] + 1, 1))(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(args.maxlen - filter_sizes[1] + 1, 1))(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(args.maxlen - filter_sizes[2] + 1, 1))(conv_2)
        maxpool_3 = MaxPool2D(pool_size=(args.maxlen - filter_sizes[3] + 1, 1))(conv_3)

        z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = Flatten()(z)
        z = Dropout(0.1)(z)

        outp = Dense(args.num_classes, activation="sigmoid")(z)

        self.model = Model(inputs=inp, outputs=outp)
