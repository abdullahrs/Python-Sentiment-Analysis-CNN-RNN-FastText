from tensorflow.python.keras.layers import Input, Embedding, Conv1D, BatchNormalization, MaxPool1D, GlobalMaxPool1D, Flatten, Dense
from tensorflow.python.keras.models import Model
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class CnnModel:
    def __init__(self, train_data, train_labels, validation_data, validation_labels, test_data, test_labels, maks_len, maks_features) -> None:
        # C'tor da gelen verileri sinif degiskenleri haline getiriyoruz
        self.train_data = train_data
        self.train_labels = train_labels

        self.validation_data = validation_data
        self.validation_labels = validation_labels

        self.test_data = test_data
        self.test_labels = test_labels

        self.maks_len = maks_len
        self.maks_features = maks_features

    def build_cnn(self) -> Model:
        sequences = Input(shape=(self.maks_len,))
        embedded = Embedding(self.maks_features, 64)(sequences)
        x = Conv1D(64, 3, activation='relu')(embedded)
        x = BatchNormalization()(x)
        x = MaxPool1D(3)(x)
        x = Conv1D(64, 5, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(5)(x)
        x = Conv1D(64, 5, activation='relu')(x)
        x = GlobalMaxPool1D()(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=sequences, outputs=outputs)
        model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

    def run_cnn(self):
        model = self.build_cnn()
        history = model.fit(
            self.train_data,
            self.train_labels,
            batch_size=128,
            epochs=3,
            validation_data=(self.validation_data, self.validation_labels),)

        preds = model.predict(self.test_data)
        print(
            f'Accuracy score: {accuracy_score(self.test_labels, 1 * (preds > 0.5))}')
        print(f'F1 score: {f1_score(self.test_labels, 1 * (preds > 0.5))}')
        print(f'ROC AUC score: {roc_auc_score(self.test_labels, preds)}')
        return history


