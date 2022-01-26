import numpy as np
import pandas as pd
import fasttext
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import bz2
import csv
from os import path

TRAIN_GENERATED_TXT_PATH = '../data/generated/train.txt'
TEST_GENERATED_TXT_PATH = '../data/generated/test.txt'


class FastTextModel:
    """
        path_list: [train_data_path, test_data_path]
    """

    def __init__(self, path_list) -> None:
        self.train_data = bz2.BZ2File(path_list[0])
        self.train_data = self.train_data.readlines()
        self.train_data = [d.decode('utf-8') for d in self.train_data]

        self.test_data = bz2.BZ2File(path_list[1])
        self.test_data = self.test_data.readlines()
        self.test_data = [d.decode('utf-8') for d in self.test_data]
        # train_supervised da kullanilan normal txt formatina cevrilen veri setleri
        if not path.exists(TRAIN_GENERATED_TXT_PATH):
            train_data = pd.DataFrame(self.train_data)
            test_data = pd.DataFrame(self.test_data)
            train_data.to_csv(TRAIN_GENERATED_TXT_PATH, index=False, sep=' ',
                                   header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
            test_data.to_csv(TEST_GENERATED_TXT_PATH, index=False, sep=' ',
                                  header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
            print("[FastTextModel][__init__] File initalization Done ✓")

    def run_fasttext(self):
        # Model train datasiyle egitiliyor
        model = fasttext.train_supervised(
            TRAIN_GENERATED_TXT_PATH, label_prefix='__label__', thread=4, epoch=10)
        print(f"[FastTextModel][run_fasttext] {model.labels} Done ✓")
        # Test datasindaki label'lar siliniyor
        new_data = [line.replace('__label__2 ', '') for line in self.test_data]
        new_data = [line.replace('__label__1 ', '') for line in new_data]
        new_data = [line.replace('\n', '') for line in new_data]
        # Test datasindaki silinmis labellara olusan yeni data'nin label'lari tahmin ediliyor
        pred = model.predict(new_data)
        # Labellar'i silinmemis test verilerindeki label'lar aliniyor
        labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in self.test_data]
        # Labellari silinmis test verileri icin tahminler label'larla eslestiriliyor
        pred_labels = [0 if x == ['__label__1'] else 1 for x in pred[0]]

        print(
            f'Accuracy score: {accuracy_score(labels, pred_labels)}')
        print(f'F1 score: {f1_score(labels, pred_labels)}')
        print(f'ROC AUC score: {roc_auc_score(labels, pred_labels)}')