from preprocessor import Preprocessor
from cnn import CnnModel
from rnn import RnnModel
from fast_text import FastTextModel


def main():
    # DATASET : https://www.kaggle.com/bittlingmayer/amazonreviews
    PATH_LIST = ['../data/train.ft.txt.bz2', 
                 '../data/test.ft.txt.bz2']
    preprocessor = Preprocessor(PATH_LIST)
    td, tl, vd, vl, ted, tel, l, f = preprocessor.get_data()
    # cnn = CnnModel(td,tl,vd,vl,ted,tel,l,f)
    # history = cnn.run_cnn()
    # preprocessor.show_results(history, metric='binary_accuracy', val_metric='val_binary_accuracy')

    rnn = RnnModel(td, tl, vd, vl, ted, tel, l, f)
    history = rnn.run_rnn()
    preprocessor.show_results(
        history, metric='binary_accuracy', val_metric='val_binary_accuracy')

    # fast_text = FastTextModel(PATH_LIST)
    # fast_text.run_fasttext()


if __name__ == "__main__":
    main()
