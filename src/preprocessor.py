import numpy as np
from bz2 import BZ2File
from re import compile
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import pickle
from os import path


class Preprocessor:
    # train ve test dosyalarinin pathlerini iceren liste aliyor
    """
        path_list: [train_data_path, test_data_path]
    """

    def __init__(self, path_list) -> None:
        self.max_feautes = 10000
        if path.exists('../data/generated/test_data.pkl'):
            with open('../data/generated/train_data.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open('../data/generated/train_labels.pkl', 'rb') as f:
                self.train_labels = pickle.load(f)
            with open('../data/generated/validation_data.pkl', 'rb') as f:
                self.validation_data = pickle.load(f)
            with open('../data/generated/validation_labels.pkl', 'rb') as f:
                self.validation_labels = pickle.load(f)
            with open('../data/generated/test_data.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
            with open('../data/generated/test_labels.pkl', 'rb') as f:
                self.test_labels = pickle.load(f)
            self.maks_len = max(len(t) for t in self.train_data)
            print(f"[Preprocessor][__init__] Values loaded ✓")
            print(f"[Preprocessor][__init__] Train data length {len(self.train_data)}")
            print(f"[Preprocessor][__init__] Train data labels length {len(self.train_labels)}")
            print(f"[Preprocessor][__init__] Validation data length {len(self.validation_data)}")
            print(f"[Preprocessor][__init__] Validation data labels length {len(self.validation_labels)}")
            print(f"[Preprocessor][__init__] Test data length {len(self.test_data)}")
            print(f"[Preprocessor][__init__] Test data labels length {len(self.test_labels)}")
            return
        # Veriler train, test olarak ayriliyor
        train_labels, train_data = self.split_data(path_list[0])
        self.test_labels, self.test_data = self.split_data(path_list[1])
        # Train ve test verilerinin textleri normalize ediliyor
        train_data = self.text_preprocessing(train_data)
        self.test_data = self.text_preprocessing(self.test_data)

        # Train datalari, train/validation olarak ikiye ayriliyor
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            train_data, train_labels, random_state=57643892, test_size=0.2)

        # Verileri tokenize ediyoruz
        self.tokenizer()
        # Batch'leri etkili kullanmak icin her bir cumleyi ayni boyuta
        # getirmek icin padding veriyoruz
        self.padding_to_sequences()
        with open('../data/generated/train_data.pkl', 'wb') as f:
            pickle.dump(self.train_data, f)
        with open('../data/generated/train_labels.pkl', 'wb') as f:
            pickle.dump(self.train_labels, f)
        with open('../data/generated/validation_data.pkl', 'wb') as f:
            pickle.dump(self.validation_data, f)
        with open('../data/generated/validation_labels.pkl', 'wb') as f:
            pickle.dump(self.validation_labels, f)
        with open('../data/generated/test_labels.pkl', 'wb') as f:
            pickle.dump(self.test_labels, f)
        with open('../data/generated/test_data.pkl', 'wb') as f:
            pickle.dump(self.test_data, f)

    def get_data(self):
        return self.train_data, self.train_labels, self.validation_data, self.validation_labels, self.test_data, self.test_labels, self.maks_len, self.max_feautes

    # Gelen bz2 dosyasinin her bir satirindaki label ve text'i ayiriyor
    def split_data(self, file):
        labels = []
        texts = []
        for line in BZ2File(file):
            x = line.decode("utf-8")
            # label'lar __label__1 ve __label__2 seklinde
            # her satirin ilk 10 karakteri label'i gerisi
            labels.append(int(x[9]) - 1)
            # text degerleri veriyor
            texts.append(x[10:].strip())
        print(f"[Preprocessor][split_data] File : {file} Done ✓")
        return (np.array(labels), texts)

    # Satirlardan gelen datalar lower case'e cevriliyor sonrasinda
    # icinde alpha-numeric ve ascii icinde bulunmayan karakterler kaldiriliyor
    def text_preprocessing(self, data):
        non_alpha_numeric = compile(r'[\W]')
        non_ascii = compile(r'[^a-z0-1\s]')
        normalized_texts = []
        for d in data:
            # tum veriler lower case'e cevriliyor
            lower = d.lower()
            # l-case'e alinan verinin icindeki noktalama isaretleri bosluk ile
            punctuation_removed = non_alpha_numeric.sub(r' ', lower)
            # noktalamalari kaldirilan verideki ascii olmayan karakterlerde silinerek
            no_non_ascii = non_ascii.sub(r'', punctuation_removed)
            # kalan veri listeye ekleniyor
            normalized_texts.append(no_non_ascii)
        print("[Preprocessor][text_preprocessing] Done ✓")
        return normalized_texts

    # Kelimeler tokenize ediliyor
    def tokenizer(self):
        tokenizer = Tokenizer(num_words=self.max_feautes)
        tokenizer.fit_on_texts(self.train_data)
        self.train_data = tokenizer.texts_to_sequences(self.train_data)
        self.validation_data = tokenizer.texts_to_sequences(
            self.validation_data)
        self.test_data = tokenizer.texts_to_sequences(self.test_data)
        print("[Preprocessor][tokenizer] Done ✓")

    # Textlere padding veriliyor
    def padding_to_sequences(self):
        self.maks_len = max(len(t) for t in self.train_data)
        # print(f"maks_len : {self.maks_len}")
        self.train_data = pad_sequences(self.train_data, maxlen=self.maks_len)
        self.validation_data = pad_sequences(
            self.validation_data, maxlen=self.maks_len)
        self.test_data = pad_sequences(self.test_data, maxlen=self.maks_len)
        print("[Preprocessor][padding_to_sequences] Done ✓")

    def show_results(self, history, metric='accuracy', val_metric='val_accuracy'):
        plt.style.use('ggplot')

        acc = history.history[metric]
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        val_acc = history.history[val_metric]

        epochs = range(len(acc))

        plt.figure(figsize=(6, 5))

        plt.plot(epochs, acc, 'r', label='training_accuracy')
        plt.plot(epochs, val_acc, 'b', label='validation_accuracy')
        plt.title('Training and Validation Accuarcy')

        plt.xlabel('-----epochs--->')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, loss, 'r', label='training_loss')
        plt.plot(epochs, val_loss, 'b', label='validation_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('----epochs--->')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
