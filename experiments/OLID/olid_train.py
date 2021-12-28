import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.OLID.olid_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel

olid_train = pd.read_csv('experiments/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('experiments/OLID/olid_test.csv', sep="\t")


olid_train, olid_validation = train_test_split(olid_train, test_size=0.2)
model = OffensiveNNModel(model_type="lstm", embedding_model_name="word2vec-google-news-300", train_df=olid_train, args=args, eval_df=olid_validation)