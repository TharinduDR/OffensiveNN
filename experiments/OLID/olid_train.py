import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.OLID.olid_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

olid_train = pd.read_csv('experiments/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('experiments/OLID/olid_test.csv', sep="\t")

olid_train['Class'] = encode(olid_train["Class"])


olid_train, olid_validation = train_test_split(olid_train, test_size=0.2)
test_sentences = olid_test['Text'].tolist()

model = OffensiveNNModel(model_type_or_path="lstm", embedding_model_name="word2vec-google-news-300", train_df=olid_train, args=args, eval_df=olid_validation)
model.train_model()

model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
predictions, raw_outputs = model.predict(test_sentences)
olid_test['predictions'] = predictions

olid_test['predictions'] = decode(olid_test['predictions'])
print_information(olid_test, "predictions", "Class")