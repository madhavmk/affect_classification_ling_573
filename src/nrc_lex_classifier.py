""" This script defines a helper class to extend the NRC lexicon emotion and valence count feature to generate emotion
and valence scores for all words in the provided text, not just those that are present in the lexicon."""

# Libraries
import numpy as np
import pandas as pd

from nrclex import NRCLex
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from typing import List

# Define class to generate NRC-lex scores
class ExtendedNRCLex():

    def __init__(self):
        self.glove_model = None
        self.embedding_classifier = None
        self.classes = []

    def fit_spanish(self, dict: dict):
        """ 
        Trains a logistic regression classifier to predict emotion and valence scores using Spanish BERT embeddings of
        words in the NRC lexicon

        Arguments:
        ----------
        dict
            A dictionary containing the Spanish NRCLex data, with phrases as keys and the emotion list as values
        """
        # Load the tokenizer and model
        model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Access the embedding layer weights
        embedding_layer = model.embeddings.word_embeddings
        embedding_weights = embedding_layer.weight.detach().numpy()

        # Build the embeddings index
        embeddings_index = {}
        vocab = tokenizer.get_vocab()

        for word, idx in vocab.items():
            embeddings_index[word] = embedding_weights[idx]

        # save the BERT model
        self.model = embeddings_index

        # Generate training data: list of NRCLex words with emotion and valence values

        lex_words = dict.keys()
        x_vec = []
        y_vec = {'fear': [], 'anger': [], 'anticipation': [], 'trust': [], 'surprise': [], 'positive': [], 'negative': [],
                 'sadness': [], 'disgust': [], 'joy': []}
        for word in lex_words:
            # check to see if 'word' is actually several words
            wordlist = word.split()

            # run through each word from the phrase, assigning the phrase weight to it
            if len(wordlist) > 1:
                for w in wordlist:
                    if w in embeddings_index.keys():
                        x_vec.append(embeddings_index[w])
                        emo_list = dict[word]
                        for emo in y_vec.keys():
                            if emo in emo_list:
                                y_vec[emo].append(1)
                            else:
                                y_vec[emo].append(0)
            else:
                if word in embeddings_index.keys():
                    x_vec.append(embeddings_index[word])
                    emo_list = dict[word]
                    for emo in y_vec.keys():
                        if emo in emo_list:
                            y_vec[emo].append(1)
                        else:
                            y_vec[emo].append(0)
        y_vec = pd.DataFrame(y_vec)
        self.classes = y_vec.columns
        x_vec = pd.DataFrame(x_vec)

        # Initialize and train classifier
        logReg = MultiOutputClassifier(LogisticRegression(penalty='l2', random_state=42, solver='sag',
                                                          multi_class='multinomial', max_iter=1000)
                                       )
        logReg.fit(x_vec, y_vec)

        # Save trained classifier
        self.embedding_classifier = logReg


    # Method to train the logistic regression
    def fit(self, embedding_file_path: str = 'data/glove.twitter.27B.25d.txt') -> None:
        """ Trains a logistic regression classifier to predict emotion and valence scores using the glove embeddings of
        words in the NRC lexicon

        Arguments:
        ----------
        embedding_file_path
            Points to the file containing the GloVe embeddings.
        """

        # Load glove embeddings
        embeddings_index = {}
        with open(embedding_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        # Save the glove model
        self.model = embeddings_index

        # Generate training data: list of NRCLex words with emotion and valence values
        lex0 = NRCLex('init')
        lex_words = list(lex0.lexicon.keys())
        x_vec = []
        y_vec = {'fear': [], 'anger': [], 'anticipation': [], 'trust': [], 'surprise': [], 'positive': [], 'negative': [],
                 'sadness': [], 'disgust': [], 'joy': []}
        for word in lex_words:
            if word in embeddings_index.keys():
                x_vec.append(embeddings_index[word])
                emo_list = NRCLex(word).affect_list
                for emo in y_vec.keys():
                    if emo in emo_list:
                        y_vec[emo].append(1)
                    else:
                        y_vec[emo].append(0)
        y_vec = pd.DataFrame(y_vec)
        self.classes = y_vec.columns
        x_vec = pd.DataFrame(x_vec)

        # # Commented out - Included to investigate results of initial classifier
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, Y_train, Y_test = train_test_split(x_vec, y_vec, test_size = 0.2, random_state = 42)
        # # Initialize and train classifier
        # logReg = MultiOutputClassifier(LogisticRegression(penalty='l2', random_state=42, solver='sag',
        #                                                   multi_class='multinomial', max_iter=1000)
        #                                )
        # logReg.fit(X_train, Y_train)
        #
        # Y_est = logReg.predict(X_test)
        # acc_dict = {}
        # i = 0
        # for emo in y_vec.columns:
        #     true_vec = Y_test[emo].values
        #     est_vec = Y_est[:,i]
        #     acc = np.sum(1.0 - np.abs(true_vec-est_vec))/len(true_vec)
        #     acc_dict[emo] = acc

        # Initialize and train classifier
        logReg = MultiOutputClassifier(LogisticRegression(penalty='l2', random_state=42, solver='sag',
                                                          multi_class='multinomial', max_iter=1000)
                                       )
        logReg.fit(x_vec, y_vec)

        # Save trained classifier
        self.embedding_classifier = logReg

    def transform(self, text: str, res_type: str = 'prob') -> List[float]:
        """ Creates emotion and valence scores for the provided word using GloVe embeddings and a trained classifier.

        Arguments:
        ----------
        text
            The text for which the scores are to be generated.
        res_type
            Indicates whether the results of the classifier prediction should be binary values ('binary') or probabilities
            ('prob').

        Returns:
        --------
            A dictionary of the emotion and valence scores for the provided text. Emotion and valence categories are keys
            that point to the associated scores for the text.
        """

        # Get embeddings
        text_vec = word_tokenize(text)
        res = np.array([0]*10)
        for word in text_vec:
            if word in self.model.keys():
                emb = self.model[word]
                emb = pd.DataFrame(emb).T
                if res_type == 'binary':
                    nrc_vals = self.embedding_classifier.predict(emb)
                else:
                    out_vals = self.embedding_classifier.predict_proba(emb)
                    nrc_vals = [x[0][1] for x in out_vals]
                    nrc_vals = np.array(nrc_vals)
                res = res + nrc_vals
        res = res / len(text_vec)

        return res


if __name__ == '__main__':

    # Initialize and train the NRC classifier
    embedding_file_path = '/Users/lindsayskinner/Documents/school/CLMS/573/data/fe_data/glove_twitter_files/glove.twitter.27B.100d.txt'
    NewNRCLex = ExtendedNRCLex()
    NewNRCLex.fit(embedding_file_path)

    text = 'this is a test run of the classification system i am so excited to test it'
    ex1 = NewNRCLex.transform(text, res_type='binary')
    ex2 = NewNRCLex.transform(text, res_type='prob')
