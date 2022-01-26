from tensorflow.python.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.python.keras.models import Model, Sequential
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class RnnModel:
    def __init__(self, train_data, train_labels, validation_data, validation_labels, test_data, test_labels, maks_len, maks_features) -> None:
        # C'tor da gelen verileri sinif degiskenleri haline getiriyoruz
        # self.train_data = train_data
        # self.train_labels = train_labels
        self.train_data = train_data[:1000000]
        self.train_labels = train_labels[:1000000]

        self.validation_data = validation_data
        self.validation_labels = validation_labels

        self.test_data = test_data
        self.test_labels = test_labels

        self.maks_len = maks_len
        self.maks_features = maks_features

    def build_rnn_model(self):
        # # GPU modeli
        # sequences = Input(shape=(self.maks_len,))
        # embedded = Embedding(self.maks_features, 64)(sequences)
        # x = CuDNNGRU(128, return_sequences=True)(embedded)
        # x = CuDNNGRU(128)(x)
        # x = LSTM(128,activation='relu')(x)
        # x = Dense(128, activation='relu')(x)
        # x = Dense(32, activation='relu')(x)
        # outputs = Dense(1, activation='sigmoid')(x)
        # model = Model(inputs=sequences, outputs=outputs)

        # # CPU ile kullanılan LSTM modeli
        sequences = Input(shape=(self.maks_len,))
        embedded = Embedding(self.maks_features, 64)(sequences)
        x = Bidirectional(LSTM(20, return_sequences=True))(embedded)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=sequences, outputs=outputs)

        model.compile(
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])

        print("[RnnModel][build_rnn_model] Done ✓")
        return model

    def run_rnn(self):
        model = self.build_rnn_model()
        history = model.fit(
            self.train_data,
            self.train_labels,
            batch_size=128,
            epochs=5,
            validation_data=(self.validation_data, self.validation_labels),)

        preds = model.predict(self.test_data)
        # Normal predictions hata veriyor prediction degerlerini normalize etmek gerekiyor
        preds = [1 * (p.mean() > 0.5) for p in preds]

        print('Accuracy score: {:0.4}'.format(accuracy_score(self.test_labels, preds)))
        print('F1 score: {:0.4}'.format(f1_score(self.test_labels, preds)))
        print('ROC AUC score: {:0.4}'.format(roc_auc_score(self.test_labels, preds)))

        return history
