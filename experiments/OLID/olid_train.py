import pandas as pd

from experiments.OLID.olid_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel

olid_train = pd.read_csv('experiments/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('experiments/OLID/olid_test.csv', sep="\t")

model = OffensiveNNModel(model_type="lstm", embedding_model_name="word2vec-google-news-300", train_df=olid_train, args=args)