"""This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import torch
import nltk
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow_hub as hub
import csv
import re

from nrc_lex_classifier import ExtendedNRCLex

from nrclex import NRCLex
# from googletrans import Translator
from translate import Translator
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel
from typing import List, Union, Optional, Dict
from copy import deepcopy
from sentence_transformers import SentenceTransformer



# Define helper function to aggregate embeddings
def get_embedding_ave(embedding_list: List[np.array], embedding_dim: int) -> np.array:
    """Function to average a list of word embeddings in order to generate a single sentence embedding.

    Arguments:
    ----------
    embedding_list
        The list of word embeddings to be averaged.
    embedding_dim
        The dimension of the embeddings.

    Returns:
    --------
        A single, aggregated embedding that is averaged over the provided list. Returns a 0 embedding when the list is
        empty.
    """
    if len(embedding_list) > 0:
        agg_embedding = sum(embedding_list) / len(embedding_list)
    else:
        agg_embedding = np.zeros(embedding_dim)

    return agg_embedding


# Define class to perform feature engineering
class FeatureEngineering:

    def __init__(self):
        """Generates features from processed data to be used in hate speech detection tasks A and B, as specified in
        SemEval 2019 task 5.

        Includes methods to generate the following features:
            * _NRC_counts --> binary classification of words across ten emotional dimensions. Raw counts are then
                                    transformed into proportions to normalize across tweets.
            * example feature2 --> fill this in with actual feature
            * example feature3 --> fill this in with actual feature
        """
        
        # Initialize the cleaned datasets
        self.train_data: Optional[pd.DataFrame] = None

        # Set fit flag
        self.fitted = False


        # Save normalization info
        self.normalization_dict = {}

        # Save embedding info
        self.embedding_file_path = None
        self.embedding_dim = None

        # Save whether the NRCLex spanish data has already been processed
        self.read_data = 0

        # save language info
        self.language = None


        # Save extended-NRC info
        self.nrc_embeddings = None
        self.nrc = None
        self.senti_dict = None
        self.word_dict = None


    def get_slang_score(self, data: pd.DataFrame, slang_dict_path: str, stop_words_path: str) -> pd.DataFrame:
        """This method uses data from the SlangSD resource, which labels slang words with their
        sentiment strength. The sentiment strength scale is from -2 to 2, where -2 is
        strongly negative, -1 is negative, 0 is neutral, 1 is positive, and 2 is strongly positive.
        This method sums the sentiment scores across all the slang words (stop words not included) in a tweet. 
        The resulting accumulated sentiment scores are added to the original dataframes as sentiment features.
        Arguments:
        ---------
        data
            The dataframe for which the slang word sentiment score feature is to be generated
        slang_dict_path
            File path for the slang dictionary file.
        stop_words_path
            File path for the stop words list file.
        Returns:
        -------
        The original dataframe with one new column that contain the accumulated sentiment scores of slang words
        for each tweet in the dataset.
        """

        # read in the slang dictionary file and construct a slang_dict
        sd_path = slang_dict_path

        with open(sd_path, 'r') as slang_dict_file:
            reader = csv.reader(slang_dict_file, delimiter='\t')
            slang_dict = {}
            for row in reader:
                slang_dict[row[0]] = row[1]
                
        # construct a stop word list to remove stop words that are irrelevant to sentiment scores
        sw_path = stop_words_path
        stop_words_lists = open(sw_path,'r').read().split('\n')

        # helper code that lists the occuring slang words in a single tweet for every tweets in the dataset
        slang_list = []
        # stores the accumulated sentiment score for a tweet, and stored
        # as a new column to the original dataframe
        slang_score_list = []

        # iterate over every tweet to get the counts and score of slang words
        for index, row in data.iterrows():
            text = row['cleaned_text']
            # helper code that generates a list containing all occuring slang
            # words in a single tweet
            occurence = []
            # Calculate the accumulated sentiment score of a tweet
            slang_score = 0

            # iterate over the slang dict to find matching slangs in a tweet
            for slang_key in list(slang_dict.keys()):
                my_regex = r"\b" + re.escape(slang_key) + r"\b"
                match_slang = re.findall(my_regex, text)
                if match_slang:
                    # only count sentiment scores for slangs that are not in the stop words list
                    if match_slang[0] not in stop_words_lists:
                        num_of_occur = len(re.findall(my_regex, text))
                        # add the sentiment score of matched slang word to slang_score
                        slang_score += num_of_occur * int(slang_dict[slang_key])
                        occurence.append(match_slang)

            slang_list.append(occurence)
            # add the slang sentiment score to slang_score_list for a tweet
            slang_score_list.append(slang_score)

        # add a new column to the original dataframe, representing the slang word sentiment score
        # for a tweet
        for index, row in data.iterrows():
            data.loc[:, "slangscore"] = slang_score_list
            # helper code to ensure all the slang words of a tweet have been successfully retrieved
            #data.loc[:, "slanglist"] = slang_list

        return data

    def _translator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method implements a translator to translate the dataframe's cleaned_text from Spanish to English.

        Arguments:
        -------
        data
            Pandas dataframe containing the preprocessed data
        
        Returns:
        -------
        The original dataset with one additional column labeled 'translated_text'
        """
        translator = Translator(from_lang="es", to_lang="en")

        def translate_tweet(tweet):

            # Try this code instead if you reach your daily allotment through the translate package, 
            # And uncomment package import at top of file
            # translator = Translator()
            # translated = translator.translate(tweet, dest='en', src='es')

            translated = translator.translate(tweet)
            return translated

        data['translated_text'] = data['cleaned_text'].apply(translate_tweet)
        return data


    def _load_translations(self, trans_path: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method reads in a .csv file with Spanish sentences translated into English,
        and appends them as a column to the dataframe

        Arguments:
        -------
        trans_path
            The path to the .csv file containing the Spanish to English translation data.
        data
            Pandas dataframe containing the preprocessed data.
        
        Returns:
        -------
        The original dataset with one additional column labeled 'translated_text'
        """
        # Load the translations file
        trans_df = pd.read_csv(trans_path)

        # Add the translations column to the dataframe
        data['translated_text'] = trans_df['translated_text'].astype(str).values

        return data

    def _Span_NRC_counts(self, span_nrc_path: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses a translated set of data from the NRC Word-Emotion Association Lexicon, which labels words with either 
        a 1 or 0 based on the presence or absence of each of the following emotional dimensions: anger, anticipation, disgust, fear, 
        joy, negative, positive, sadness, surprise, trust. It sums the frequency counts in each of the ten dimensions across all the words 
        in a tweet, then divides by the total number of counts to obtain a proportion. These proportions are added on to the end 
        of the dataframe as count-based features.

        Arguments:
        ---------
        span_nrc_path
            The path to the .csv file containing the Spanish NRCLex data
        data
            The data for which the feature is to be generated

        Returns:
        -------
        The original dataset with ten new columns that contain the new emotion features generated for each tweet in the
        dataset.

        """
        if self.read_data == 0:
            with open(span_nrc_path, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                senti_dict = {}
                word_dict = {}
                for row in reader:
                    s_word = row[0]
                    sentiment = row[1]
                    e_word = row[2]

                    # create dictionary mapping spanish word/phrase to English word/phrase
                    # ensure no duplicates
                    if s_word in word_dict:
                        translations = word_dict[s_word]
                        if e_word in translations:
                            pass
                        else:
                            word_dict[s_word].append(e_word)
                    else:
                        word_dict[s_word] = [e_word]

                    # create dictionary that catalogues the sentiments attached to a spanish phrase
                    # aggregate over all phrase translations
                    if s_word in senti_dict:
                        sentiments = senti_dict[s_word]
                        if sentiment in sentiments:
                            pass
                        else:
                            senti_dict[s_word].append(sentiment)
                    else:
                        senti_dict[s_word] = [sentiment]
            
                self.senti_dict = senti_dict
                self.word_dict = word_dict
                self.read_data = 1
                
        # add ten columns to the end of the dataframe, representing the eight emotional dimensions of NRC
        emotions = ['negative_esp', 'positive_esp', 'anger_esp', 'anticipation_esp', 'disgust_esp', 'fear_esp', 'joy_esp', 'sadness_esp', 'surprise_esp', 'trust_esp']
        for emotion in emotions:
            data[emotion] = 0

        # load spanish tokenizer from nltk
        tokenize_spanish = nltk.data.load('tokenizers/punkt/PY3/spanish.pickle')

        # iterate over each tweet to get counts of each emotion classification
        for index, row in data.iterrows():
            text = tokenize_spanish.tokenize(row['cleaned_text'])

            # iterate over each word in the tweet, add counts to emotion vector
            for word in text:
                if word in self.senti_dict:
                    emolist = self.senti_dict[word]
                    for emo in emolist:
                        label = emo + "_esp"
                        data.at[index, label] += 1

        # divide by total count of emo markers to get proportions not frequency counts
        # Replace 0 values with NaN to prevent error with dividing by zero
        rowsums = data.iloc[:, -10:].sum(axis=1)
        rowsums[rowsums == 0] = 1.0
        data.iloc[:, -10:] = data.iloc[:, -10:].div(rowsums, axis=0)

        return data

    def _NRC_counts(self, data: pd.DataFrame) -> pd.DataFrame:

        """This method uses data from the NRC Word-Emotion Association Lexicon, which labels words with either a 1 or 0 based on
        the presence or absence of each of the following emotional dimensions: anger, anticipation, disgust, fear, joy, negative, 
        positive, sadness, surprise, trust. It sums the frequency counts in each of the ten dimensions across all the words 
        in a tweet, then divides by the total number of counts to obtain a proportion. These proportions are added on to the end 
        of the dataframe as count-based features.


        Arguments:
        ---------
        data
            The data for which the feature is to be generated

        Returns:
        -------
        The original dataset with ten new columns that contain the new emotion features generated for each tweet in the
        dataset.

        """
        # add ten columns to the end of the dataframe, representing the eight emotional dimensions of NRC
        emotions = ['negative', 'positive', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        for emotion in emotions:
            data[emotion] = 0.0
        
        # iterate over each tweet to get counts of each emotion classification
        for index, row in data.iterrows():
            if row['cleaned_text'] != '':
                if self.language == 'english':
                    text = word_tokenize(row['cleaned_text'])
                elif self.language == 'spanish':
                    text = word_tokenize(row['translated_text'])

                # iterate over each word in the tweet, add counts to emotion vector
                for word in text:
                    emotion = NRCLex(word)
                    if emotion is not None:
                        emolist = emotion.affect_list
                        for emo in emolist:
                            data.at[index, emo] += 1

        # divide by total count of emo markers to get proportions not frequency counts
        # Replace 0 values with NaN to prevent error with dividing by zero
        rowsums = data.iloc[:, -10:].sum(axis=1)
        rowsums[rowsums == 0] = 1.0
        data.iloc[:, -10:] = data.iloc[:, -10:].div(rowsums, axis=0)

        # ***Uncomment the line below to create file showing the data visualized***
        # data.to_csv('test.txt', sep=',', header=True)

        return data

    def _extended_NRC_counts(self, data: pd.DataFrame, embedding_file: str):
        """This method uses GloVe embeddings and data from the NRC Word-Emotion Association Lexicon, which labels words
        with either a 1 or 0 based on the presence or absence of each of the following emotional dimensions: anger,
        anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust. A classification model is
        trained on GloVe embeddings to predict the affiliated NRC emotion and valence values. This extends the basic
        NRC counts to provide counts for any word/sub-word for which one can generate a GloVe embedding. The
        probabilities predicted for each category are summed across each of the ten dimensions, across all the words in
        a tweet, then divides by the total number of words. These proportions are added on to the end of the dataframe.


        Arguments:
        ---------
        data
            The data for which the feature is to be generated
        embeddings_file
            Points to the file containing the GloVe embeddings.

        Returns:
        -------
        The original dataset with ten new columns that contain the new emotion features generated for each tweet in the
        dataset.
        """

        # Initialize and train the NRC classifier
        if not self.fitted:
            if self.language == "english":
                NewNRCLex = ExtendedNRCLex()
                NewNRCLex.fit(embedding_file)
                self.nrc = NewNRCLex
            elif self.language == "spanish":
                NewNRCLex = ExtendedNRCLex()
                NewNRCLex.fit_spanish(self.senti_dict)
                self.nrc = NewNRCLex
        else:
            NewNRCLex = self.nrc
        emo_classes = NewNRCLex.classes
        emo_classes = emo_classes + '_ext'

        # iterate over each tweet to get counts of each emotion classification
        for emo in emo_classes:
            data[emo] = 0.0
        for index, row in data.iterrows():
            if row['cleaned_text'] != '':
                text = row['cleaned_text']
                if len(text) > 0:
                    res = NewNRCLex.transform(text, res_type='prob')
                    for i in range(len(res)):
                        emo = emo_classes[i]
                        data.at[index, emo] = res[i]

        return data

    def get_es_sent_score(self, data: pd.DataFrame, es_sent_path: str) -> pd.DataFrame:
        """This method uses data from the SpanishSentimentLexicons resource, which labels Spanish 
        polarity words with their sentiment score. We convert polarity words with both negative and
        positive annotations into neutral words. The resulting sentiment evaluation is a trinary 
        sentiment scale, where -1 is negative, 0 is neutral, and 1 is positive. This method sums the
        sentiment scores across all the polarity words in a tweet. The resulting accumulated sentiment 
        scores are added to the original dataframes as sentiment features.


        Arguments:
        ---------
        data
            The dataframe for which the Spanish word sentiment score feature is to be generated
        es_sent_path
            File path for the Spanish sentiment score file.

        Returns:
        -------
        The original datasframe with one new column that contain the accumulated sentiment scores of 
        polarity words for each tweet in the dataset.

        """
        
        # read in the Spanish words sentiment file
        sent_path = es_sent_path
        
        es_polar_list = []
        raw_es_polar = open(sent_path,'r').readlines()
        
        # construct a trinary_es_dict (Spanish words trinary sentiment dict)
        for e in raw_es_polar:
            # raw input with undetermined polarity entries
            temp = e.strip().split('\t')
            del temp[1]
            # indicates this entry has mixed polarity
            if len(temp) != 2:
                # indicates the polarity of this entry is determined
                if temp[1] == temp[2]:
                    temp = temp[:2]
                    if temp[1] == 'pos':
                        word_polar = (temp[0], '1')
                        es_polar_list.append(word_polar)
                    else:
                        word_polar = (temp[0], '-1')
                        es_polar_list.append(word_polar)
                # indicates this entry has conflict sentiments and set it to neutral
                else:
                    word_polar = (temp[0], '0')
                    es_polar_list.append(word_polar)  
            else:
                if temp[1] == 'pos':
                    word_polar = (temp[0], '1')
                    es_polar_list.append(word_polar)
                else:
                    word_polar = (temp[0], '-1')
                    es_polar_list.append(word_polar)

        trinary_es_dict = dict(es_polar_list)
            
        # helper code that lists the occuring polarity words in a single tweet
        sent_word_list = []  
        # stores the accumulated sentiment score for a tweet, and stored
        # as a new column to the original dataframe
        sent_score_list = []     
                
        # iterate over every tweet to get the counts and score of polarity words
        for index, row in data.iterrows(): 
            text = row['cleaned_text']
            # helper code that generates a list containing all occuring polarity
            # words in a single tweet
            # ***Uncomment the line below to show all the occuring polarity words
            occurence = [] 
            # Calculate the accumulated sentiment score of a tweet
            sent_score = 0
            
            # iterate over the trinary_es_dict to find matching sentiment triggering words in a tweet
            for sent_key in list(trinary_es_dict.keys()):  
                my_regex = r"\b" + re.escape(sent_key) + r"\b"
                match_sent_words = re.findall(my_regex, text)
                if match_sent_words:
                    num_of_occur = len(re.findall(my_regex, text))
                    # add the sentiment score of sentiment triggering words to sent_score
                    sent_score += num_of_occur * int(trinary_es_dict[sent_key])
                    occurence.append(match_sent_words)
                
            sent_word_list.append(occurence)
            # add the polarity words' sentiment score to sent_score_list for a tweet
            sent_score_list.append(sent_score)
            
        # add a new column to the original dataframe, representing the polarity words' sentiment score
        # for a tweet
        for index, row in data.iterrows():
            # ***Uncomment the line below to show all the occuring polarity words
            # for a tweet as a column of dataframe 
            #data.loc[:, "sentwordlist"] = sent_word_list
            data.loc[:, "sent_score"] = sent_score_list
            
        return data


    def embeddings_helper(self, tweet: str, model: Union[Dict, KeyedVectors, PreTrainedModel], embedding_type: str,
                          tokenizer: Optional[PreTrainedTokenizerBase] = None) -> List[List[float]]:
        """Helper function to get FastText, BERTweet, or GloVe embeddings. Tokenizes input and accesses embeddings
        from model/dictionary.

        Arguments:
        ---------
        tweet
            The line of the data to generate embeddings for
        model
            The dictionary of FastText embeddings, as either a Dict or an instance of KeyedVectors from gensim
        embedding_type
            '1' == FastText
            '2' == BERTweet
            '3' == GloVe
            '4' (or else) TwHIN-BERT for spanish
        tokenizer
            Optional Tokenizer for BERTweet embeddings

        Returns:
        -------
        A list of the word embeddings for each word in the input tweet.

        """
        # tokenize
        words = tweet.split()

        # retrieve embeddings if in the vocabulary/model
        if embedding_type == '1':
            embeddings = [model[word] for word in words if word in model.key_to_index]
        elif embedding_type == '2':

            # different form of tokenizing
            input_ids = torch.tensor([tokenizer.encode(tweet, padding=True, truncation=True, max_length=130)])
            with torch.no_grad():
                outputs = model(input_ids)

            # tokens = tokenizer.tokenize(tweet)
            # input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # with torch.no_grad():
            #     outputs = model(torch.tensor([input_ids], dtype=torch.long))

            embed = outputs.last_hidden_state[0]
            embed_np = embed.detach().numpy()
            # embeddings = [embed_np[i].tolist() for i in range(len(tokens))]
            # embeddings = [embed_np[i].tolist() for i in range(len(input_ids[0]))]
            # embeddings = np.array(embeddings).flatten()
            embeddings = np.mean(embed_np, axis=0)
        elif embedding_type == '3':
            embeddings = [model[word] for word in words if word in model.keys()]
        else:
            inputs = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)

            # Step 5: Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state

            # Step 6: Process embeddings
            # Typically, you may want to use the embeddings of the [CLS] token (first token) for each sequence
            embeddings = embeddings[:, 0, :]
            embeddings = embeddings.numpy()
            embeddings = embeddings.flatten()
            #embeddings = embeddings.tolist()

        return embeddings

    def get_fasttext_embeddings(self, df: pd.DataFrame, embedding_file_path: str):
        """Function to get FastText embeddings from a dataframe and automatically add them to this dataframe. These
        are pretrained embeddings with d_e == 300

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data
        embedding_file_path
            File path for the embeddings file

        Returns:
        -------
        Nothing

        """
        # get the model from a preloaded corpus
        model = KeyedVectors.load_word2vec_format(embedding_file_path)

        # get the embeddings for each row and save to a new column in the dataframe
        df['fastText_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, model, '1'))

    def get_bertweet_embeddings(self, df: pd.DataFrame, language: str):
        """Function to get BERTweet embeddings from a dataframe and automatically add them to this dataframe.
        These embeddings are learned from a model, with d_e == 768

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # # OLD METHOD REMOVED
        # if language == 'en':
        #     # load tokenizer and model
        #     model = AutoModel.from_pretrained("vinai/bertweet-base")
        #     tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        #     # get the embeddings for each row and save to a new column in the dataframe
        #     df['BERTweet_embeddings'] = df['raw_text'].apply(lambda tweet: self.embeddings_helper(tweet, model,
        #                                                                                               '2',
        #                                                                                               tokenizer))
        # else:
        #     tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
        #     model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
        #     # get the embeddings for each row and save to a new column in the dataframe
        #     df['BERTweet_embeddings'] = df['raw_text'].apply(lambda tweet: self.embeddings_helper(tweet, model,
        #                                                                                               '4',
        #                                                                                               tokenizer))

        tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
        model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
        # get the embeddings for each row and save to a new column in the dataframe
        df['BERTweet_embeddings'] = df['raw_text'].apply(lambda tweet: self.embeddings_helper(tweet, model,
                                                                                              '4',
                                                                                              tokenizer))

    def get_glove_embeddings(self, df: pd.DataFrame, embedding_file_path: str):
        """Function to get GloVe embeddings from a dataframe and automatically add them to this dataframe. These
        are pretrained embeddings with d_e == 300. NOTE: these can work for english and spanish embeddings. If you want
        to do a certain language you only need to change the embedding_file_path in config.json to be the respective
        English or Spanish GloVe file

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data
        embedding_file_path
            File path for the embeddings file

        Returns:
        -------
        Nothing

        """
        # load embeddings and make a dict
        embeddings_index = {}
        with open(embedding_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        # get the embeddings for each row and save to a new column in the dataframe
        df['GloVe_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, embeddings_index, '3'))

    def get_universal_sent_embeddings(self, df: pd.DataFrame, language: str):
        """Function to get Google Universal Sentence Encoder embeddings from a dataframe and automatically add them
        to this dataframe. These embeddings are for a whole sentence rather than for individual words and are of
        d_e == 512. Is also now able to do Spanish sentences through the language argument

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # load the embeddings from tensorflow hub
        if language == 'en':
            #embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
            embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
        else:
            embed = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

        # function to reformat cleaned text for proper embedding
        def embed_text(text):
            if language == 'en':
                embeddings= embed([text])
            else:
                embeddings = embed.encode(text)
            embeddings_flat = np.array(embeddings).flatten()
            return embeddings_flat

        
        # get the embeddings for each row and save to a new column in the dataframe
        df['Universal_Sentence_Encoder_embeddings'] = df['cleaned_text'].apply(embed_text)

    def normalize_feature(self, data: pd.DataFrame, feature_columns: List[str],
                          normalization_method: Optional[str] = None) -> pd.DataFrame:
        """Normalizes the features in the specified columns by transforming the data to fall within [0,1].

        This can be done using a number of different approaches. The specific approach and relevant parameters needed to
        perform normalization in downstream transformations are saved in the normalization_dict, which is keyed by the
        feature column name.

        Arguments:
        ----------
        data
            The data for which the feature is to be generated
        feature_columns
            The column name(s) of the feature(s) to be normalized. If multiple column names are provided then the values
            in both columns are simultaneously normalized.
        normalization_method
            Required when fitting. Specifies which calculation to use to normalize the features. Options include...
            min_max:
                Applies (x-min)/(max-min) transformation, capping the resulting values to fall within [0,1].
            z_score:
                Applies Norm.CDF((x-mu)/sigma) transformation. Values correspond to percentages from a normal
                distribution.

        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the normalized features for each observation
            in the dataset.
        """

        # Initialize dictionary to hold normalized results
        normalized_feats = {}

        # Perform normalization transformations, assuming fitting has already occurred
        if self.fitted:
            for feat in feature_columns:

                # If trained normalization method uses min-max approach
                if self.normalization_dict[feat]['method'] == 'min_max':
                    f_min = self.normalization_dict[feat]['params']['min']
                    f_max = self.normalization_dict[feat]['params']['max']
                    feat_vals = data[feat]
                    norm_vals = (feat_vals - f_min) / (f_max - f_min)
                    # Cap any extreme values that fall outside the range seen in the training data
                    norm_vals[norm_vals > 1.0] = 1.0
                    norm_vals[norm_vals < 0.0] = 0.0

                # If trained normalization method uses z-score approach
                if self.normalization_dict[feat]['method'] == 'z_score':
                    sigma = self.normalization_dict[feat]['params']['sigma']
                    mu = self.normalization_dict[feat]['params']['mu']
                    feat_vals = data[feat]
                    z_scores = (feat_vals - mu) / sigma
                    norm_vals = st.norm.cdf(z_scores)

                # Store results to be returned
                normalized_feats[feat] = norm_vals

        # Learn and apply normalization transformations
        else:
            for feat in feature_columns:
                self.normalization_dict[feat] = {}
                feat_vals = data[feat]

                # If specified normalization method is min-max approach
                if normalization_method == 'min_max':
                    f_min = feat_vals.min()
                    f_max = feat_vals.max()
                    norm_vals = (feat_vals - f_min) / (f_max - f_min)
                    # Save parameters for future transformations
                    self.normalization_dict[feat]['method'] = 'min_max'
                    self.normalization_dict[feat]['params'] = {'min': f_min, 'max': f_max}

                # If specified normalization method is z-score approach
                if normalization_method == 'z_score':
                    sigma = feat_vals.std()
                    mu = feat_vals.mean()
                    z_scores = (feat_vals - mu) / sigma
                    norm_vals = st.norm.cdf(z_scores)
                    # Save parameters for future transformations
                    self.normalization_dict[feat]['method'] = 'z_score'
                    self.normalization_dict[feat]['params'] = {'sigma': sigma, 'mu': mu}

                # Store results to be returned
                normalized_feats[feat] = norm_vals

        # Add normalized features to dataframe
        n_cols = len(data.columns)
        for k in normalized_feats.keys():
            data.insert(loc=n_cols, column=f'{k}_normalized', value=normalized_feats[k])
            n_cols += 1

        return data

    def fit_transform(self, train_data: pd.DataFrame, embedding_file_path: str, embedding_dim: int,  
                      nrc_embedding_file: str, slang_dict_path: str, stop_words_path: str, es_sent_path:str, 
                      language: str, lexpath: str, load_translations: str, trans_path: Optional[str]) -> pd.DataFrame:

        """Learns all necessary information from the provided training data in order to generate the complete set of
        features to be fed into the classification model. In the fitting process, the training data is also transformed
        into the feature-set expected by the model and returned.

        Arguments:
        ---------
        train_data
            The training data that is used to define the feature-engineering methods.
        embedding_file_path
            File path for the Glove embeddings file.
        embedding_dim
            The dimension of the embeddings.
        slang_dict_path
            File path for the Slang dictionary file.
        es_sent_path
            File path for the Spanish sentiment words file.
        language
            The language of the data.
        lexpath
            The path to the Spanish NRCLex .csv file.
        load_translations
            string to load translations or save them.
        trans_path
            File path for the Spanish to English translations file.
        stop_words_path
            File path for the stop words list file.


        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the calculated features for each observation
            in the dataset.
        """
        # Save the language and lexpath variables to the model
        self.language = language
        self.lexpath = lexpath


        # Get the training data, to be used for fitting
        self.train_data = train_data
        self.train_data['cleaned_text'].fillna('', inplace=True)

        # Save the slang dictionary path for use in the model
        self.slang_dict_path = slang_dict_path
        
        # Save the stop words list path for use in the model
        self.stop_words_path = stop_words_path
        
        # Save the Spanish sentiment words file for use in the model
        self.es_sent_path = es_sent_path

        # Save language
        self.language = language

        # Normalize count features from data cleaning process
        transformed_data = self.normalize_feature(data=self.train_data,
                                                  feature_columns=['!_count', '?_count', '$_count', '*_count'],
                                                  normalization_method='z_score')

        # Get slang words sentiment scores feature
        transformed_data = self.get_slang_score(transformed_data, self.slang_dict_path, self.stop_words_path)
        
        # Get Spanish words sentiment scores feature
        if language == 'spanish':
            transformed_data = self.get_es_sent_score(transformed_data, self.es_sent_path)

        # Get NRC (emotion and sentiment word) counts feature
        if language == 'english':
            transformed_data = self._NRC_counts(transformed_data)
            self.nrc_embeddings = nrc_embedding_file
            transformed_data = self._extended_NRC_counts(transformed_data, embedding_file=nrc_embedding_file)

        elif language == 'spanish':

            # uses Spanish translated NRCLex to get counts
            transformed_data = self._Span_NRC_counts(lexpath, transformed_data)
            transformed_data = self._extended_NRC_counts(transformed_data, embedding_file=nrc_embedding_file)
            
            if load_translations == 'load':
                transformed_data = self._load_translations(trans_path, transformed_data)
                transformed_data = self._NRC_counts(transformed_data)

            else:
                # translates the cleaned text to English, runs normal NRCLex
                transformed_data = self._translator(transformed_data)
                transformed_data.to_csv('data/translations.csv')
                transformed_data = self._NRC_counts(transformed_data)
            
                
        # Get Universal Sentence embeddings
        self.get_universal_sent_embeddings(transformed_data, language)

        # Get BERTweet Sentence embeddings
        self.get_bertweet_embeddings(transformed_data, language)

        # Get Glove embeddings and aggregate across all words
        self.embedding_file_path = embedding_file_path
        self.embedding_dim = embedding_dim
        self.get_glove_embeddings(transformed_data, embedding_file_path=embedding_file_path)
        transformed_data['Aggregate_embeddings'] = transformed_data['GloVe_embeddings'].apply(
            lambda x: get_embedding_ave(x, embedding_dim))

        # Update the fitted flag
        self.fitted = True

        return transformed_data


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        """Uses the feature-generating methods that were fit in an earlier step to transform a new dataset to include
        the feature-set expected by the classification model.

        Arguments:
        ---------
        data
            The data set for which the feature set is to be generated.

        Returns:
        -------
        transformed_data
            The original dataframe with new columns that include the calculated features for each observation in the
            dataset.
        """

        # Ensure feature generating methods have been trained prior to transforming the data
        assert self.fitted, 'Must apply fit_transform to training data before other datasets can be transformed.'

        # Normalize count features from data cleaning process
        transformed_data = self.normalize_feature(data=data,
                                                  feature_columns=['!_count', '?_count', '$_count', '*_count'])

        # Get slang words sentiment scores feature
        transformed_data = self.get_slang_score(transformed_data, self.slang_dict_path, self.stop_words_path)
        
        # Get Spanish words sentiment scores feature
        if self.language == 'spanish':
            transformed_data = self.get_es_sent_score(transformed_data, self.es_sent_path)

        # Get NRC (emotion and sentiment word) counts feature
        if self.language == 'english':
            transformed_data = self._NRC_counts(transformed_data)
            transformed_data = self._extended_NRC_counts(transformed_data, embedding_file=self.nrc_embeddings)
        elif self.language == 'spanish':

            # uses Spanish translated NRCLex to get counts
            transformed_data = self._Span_NRC_counts(self.lexpath, transformed_data)
            transformed_data = self._extended_NRC_counts(transformed_data, embedding_file=self.nrc_embeddings)

            # translates the cleaned text to English, runs normal NRCLex
            transformed_data = self._translator(transformed_data)
            transformed_data = self._NRC_counts(transformed_data)
            

        # Get Universal Sentence embeddings
        self.get_universal_sent_embeddings(transformed_data, language=self.language)

        # Get BERTweet Sentence embeddings
        self.get_bertweet_embeddings(transformed_data, language=self.language)

        # Get Glove embeddings and aggregate across all words
        self.get_glove_embeddings(transformed_data, embedding_file_path=self.embedding_file_path)
        transformed_data['Aggregate_embeddings'] = transformed_data['GloVe_embeddings'].apply(
            lambda x: get_embedding_ave(x, self.embedding_dim))


        return transformed_data


if __name__ == '__main__':

    # Imports
    from data_processor import DataProcessor

    # Load and clean the raw data
    myDP = DataProcessor()
    myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac
    myDP.clean_data()

    # Instantiate the FeatureEngineering object
    myFE = FeatureEngineering()

    # Fit
    train_df = myFE.fit_transform(myDP.processed_data['train'], embedding_file_path='data/glove.twitter.27B.25d.txt',
                                embedding_dim=25, nrc_embedding_file='data/glove.twitter.27B.25d.txt',
                                slang_dict_path='data/SlangSD.txt', stop_words_path='data/stopwords.txt', es_sent_path='data/es_sent.txt', language='en')
    # Note that the embedding file is too large to add to the repository, so you will need to specify the path on your
    # local machine to run this portion of the system.

    # Transform
    val_df = myFE.transform(myDP.processed_data['validation'])

    # Save results
    import pickle as pkl
    dir_path = '../data/processed_data/D4'

    # Pickle the pre-processed training data to load in future runs
    train_data_file = f"{dir_path}/train_df.pkl"
    with open(train_data_file, 'wb') as f:
        pkl.dump(train_df, f)

    # Pickle the pre-processed validation data to load in future runs
    val_data_file = f"{dir_path}/val_df.pkl"
    with open(val_data_file, 'wb') as f:
        pkl.dump(val_df, f)
