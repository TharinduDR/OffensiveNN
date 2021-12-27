import logging
import gensim.downloader as api

import numpy as np
from keras_preprocessing.text import Tokenizer

from offensive_nn.model_args import ModelArgs
from offensive_nn.models.offensive_capsule_model import OffensiveCapsuleModel
from offensive_nn.models.offensive_cnn_model import OffensiveCNNModel
from offensive_nn.models.offensive_lstm_model import OffensiveLSTMModel

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

logger = logging.getLogger(__name__)


class OffensiveNNModel:
    def __init__(self, model_type,
                 embedding_model_name=None,
                 train_df=None,
                 num_labels=None,
                 args=None,
                 use_cuda=True,
                 cuda_device=-1,
                 **kwargs, ):

        self.train_df = train_df

        self.args = self._load_model_args(model_type)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ModelArgs):
            self.args = args

        X = self.train_df["Text"].values
        self.embedding_model = api.load(embedding_model_name)

        # print(self.embedding_model_path)

        self.tokenizer = Tokenizer(num_words=self.args.max_features, filters='')
        self.tokenizer.fit_on_texts(list(X))
        X = self.tokenizer.texts_to_sequences(X)

        self.word_index = self.tokenizer.word_index
        self.args.max_features = len(self.word_index) + 1

        self.embedding_matrix = self.get_emb_matrix(self.word_index, self.args.max_features, self.embedding_model)

        MODEL_CLASSES = {
            "cnn": OffensiveCNNModel,
            "lstm": OffensiveLSTMModel,
            "capsule": OffensiveCapsuleModel
        }

        self.nnmodel = MODEL_CLASSES[model_type](self.args, self.embedding_matrix)
        self.nnmodel.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        logger.info(self.nnmodel.model.summary())



    # def train_model(self, train_df,
    #                 multi_label=False,
    #                 output_dir=None,
    #                 show_running_loss=True,
    #                 args=None,
    #                 eval_df=None,
    #                 verbose=True,
    #                 **kwargs):



    @staticmethod
    def _load_model_args(input_dir):
        args = ModelArgs()
        args.load(input_dir)
        return args

    # @staticmethod
    # def load_word_emb(word_index, embedding_file):
    #     def get_coefs(word, *arr):
    #         return word, np.asarray(arr, dtype='float32')
    #
    #     embeddings_index = dict(
    #         get_coefs(*o.rstrip().split(" ")) for o in open(embedding_file, encoding="utf8") if
    #         o.rstrip().split(" ")[0] in word_index)
    #     return embeddings_index

    @staticmethod
    def load_word_emb(word_index, emebdding_model):
        embeddings_index = dict()
        for idx, key in enumerate(emebdding_model.key_to_index):
            if key in word_index:
                embeddings_index[key] = emebdding_model[key]
        return embeddings_index

    def get_emb_matrix(self, word_index, max_features, embedding_file):
        embeddings_index = self.load_word_emb(word_index, embedding_file)
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
        embedding_count = 0
        no_embedding_count = 0
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                embedding_count = embedding_count + 1
            else:
                no_embedding_count = no_embedding_count + 1

        no_embedding_rate = no_embedding_count/ (embedding_count + no_embedding_count)
        logger.warning("Embeddings are not found for {:.2f}% words.".format(no_embedding_rate*100))


        return embedding_matrix
