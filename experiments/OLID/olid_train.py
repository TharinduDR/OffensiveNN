import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.OLID.olid_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

import numpy as np

olid_train = pd.read_csv('experiments/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('experiments/OLID/olid_test.csv', sep="\t")

olid_train['Class'] = encode(olid_train["Class"])
test_sentences = olid_test['Text'].tolist()

test_preds = np.zeros((len(olid_test), args["n_fold"]))

for i in range(args["n_fold"]):
    olid_train, olid_validation = train_test_split(olid_train, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(model_type_or_path="cnn2D", embedding_model_name="word2vec-google-news-300", train_df=olid_train, args=args, eval_df=olid_validation)
    model.train_model()
    print("Finished Training")
    model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
    predictions, raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))


final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

olid_test['predictions'] = final_predictions

olid_test['predictions'] = decode(olid_test['predictions'])
print_information(olid_test, "predictions", "Class")