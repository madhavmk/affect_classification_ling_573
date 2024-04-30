"""This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import torch
import numpy as np
import pandas as pd
import csv
import scipy.stats as st
import tensorflow_hub as hub

from nrclex import NRCLex
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel
from typing import List, Union, Optional, Dict

# for Universal Sentence Encoder -- need to add tensorflow to environment.yml file
#import tensorflow as tf
#import tensorflow_hub as hub

# Specify the file path to the slang dictionary 
slang_dict_file_path = 'data/SlangSD.txt'

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
        
    def get_slang_score(self, data: pd.DataFrame, slang_dict_file_path: str) -> pd.DataFrame:
        """This method uses data from the SlangSD resource, which labels slang words with their  
        sentiment strength. The sentiment strength scale is from -2 to 2, where -2 is
        strongly negative, -1 is negative, 0 is neutral, 1 is positive, and 2 is strongly positive.
        This method sums the sentiment scores across all the slang words in a tweet. The resulting
        accumulated sentiment scores are added to the original dataframes as sentiment features.


        Arguments:
        ---------
        data
            The dataframe for which the slang word sentiment score feature is to be generated
        slang_dict_file_path
            File path for the slang dictionary file.

        Returns:
        -------
        The original datasframe with one new column that contain the accumulated sentiment scores of slang words 
        for each tweet in the dataset.

        """
        
        # read in the slang dictionary file and construct a slang_dict
        sd_path = slang_dict_file_path
        
        with open (sd_path, 'r') as slang_dict_file:
            reader = csv.reader(slang_dict_file, delimiter='\t')
            slang_dict = {}
            for row in reader:
                slang_dict[row[0]] = row[1]
        
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
            data[emotion] = 0
        
        # iterate over each tweet to get counts of each emotion classification
        for index, row in data.iterrows():
            text = word_tokenize(row['cleaned_text'])

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
            '3' (or anything else) == GloVe
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


            input_ids = torch.tensor([tokenizer.encode(tweet)])
            with torch.no_grad():
                outputs = model(input_ids)

            # tokens = tokenizer.tokenize(tweet)
            # input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # with torch.no_grad():
            #     outputs = model(torch.tensor([input_ids], dtype=torch.long))

            embed = outputs.last_hidden_state[0]
            embed_np = embed.detach().numpy()
            # embeddings = [embed_np[i].tolist() for i in range(len(tokens))]
            embeddings = [embed_np[i].tolist() for i in range(len(input_ids[0]))]
            embeddings = np.array(embeddings).flatten()
        else:
            embeddings = [model[word] for word in words if word in model.keys()]

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

    def get_bertweet_embeddings(self, df: pd.DataFrame):
        """Function to get BERTweet embeddings from a dataframe and automatically add them to this dataframe.
        These embeddings are learned from a model, with d_e == ?? (probably should know this)

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # load tokenizer and model
        model = AutoModel.from_pretrained("vinai/bertweet-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

        # get the embeddings for each row and save to a new column in the dataframe
        df['BERTweet_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, model, '2', tokenizer))

    def get_glove_embeddings(self, df: pd.DataFrame, embedding_file_path: str):
        """Function to get GloVe embeddings from a dataframe and automatically add them to this dataframe. These
        are pretrained embeddings with d_e == 300.

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

    def get_universal_sent_embeddings(self, df: pd.DataFrame):
        """Function to get Google Universal Sentence Encoder embeddings from a dataframe and automatically add them
        to this dataframe. These embeddings are for a whole sentence rather than for individual words and are of
        d_e == 512.

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # load the embeddings from tensorflow hub
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

        # function to reformat cleaned text for proper embedding
        def embed_text(text):
            embeddings= embed([text])
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
                #TODO: extend this method to use a more appropriate distribution than normal for certain features

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

    def fit_transform(self, train_data: pd.DataFrame, embedding_file_path: str, embedding_dim: int) -> pd.DataFrame:
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

        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the calculated features for each observation
            in the dataset.
        """

        # Get the training data, to be used for fitting
        self.train_data = train_data


        # Normalize count features from data cleaning process
        transformed_data = self.normalize_feature(data=train_data,
                                                  feature_columns=['!_count', '?_count', '$_count', '*_count'],
                                                  normalization_method='z_score')

        # Get slang words sentiment scores feature
        transformed_data = self.get_slang_score(transformed_data, slang_dict_file_path)

        # Get NRC (emotion and sentiment word) counts feature
        transformed_data = self._NRC_counts(transformed_data)
        
        # Get Universal Sentence embeddings
        self.get_universal_sent_embeddings(transformed_data)

        # Get BERTweet Sentence embeddings
        # self.get_bertweet_embeddings(transformed_data)

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
        transformed_data = self.get_slang_score(transformed_data, slang_dict_file_path)
        
        # Get NRC values
        transformed_data = self._NRC_counts(transformed_data)

        # Get Universal Sentence embeddings
        self.get_universal_sent_embeddings(transformed_data)

        # Get BERTweet Sentence embeddings
        # self.get_bertweet_embeddings(transformed_data)

        # Get Glove embeddings and aggregate across all words
        self.get_glove_embeddings(transformed_data, embedding_file_path=self.embedding_file_path)
        transformed_data['Aggregate_embeddings'] = transformed_data['GloVe_embeddings'].apply(
            lambda x: get_embedding_ave(x, self.embedding_dim))


        return transformed_data

''' The lines below are commented out so that all code is run through main.py
'''
# if __name__ == '__main__':

#     # Imports
#     from data_processor import DataProcessor


#     # Load and clean the raw data
#     myDP = DataProcessor()
#     myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac
#     myDP.clean_data()


#     # Instantiate the FeatureEngineering object
#     myFE = FeatureEngineering()

    #   # Fit
    #   train_df = myFE.fit_transform(myDP.processed_data['train'], embedding_file_path='data/glove.twitter.27B.25d.txt',
    #                                 embedding_dim=25)
    #   # Note that the embedding file is too large to add to the repository, so you will need to specify the path on your
    #   # local machine to run this portion of the system.

#     # Transform
#     val_df = myFE.transform(myDP.processed_data['validation'])

#     # View a sample of the results
#     train_df.head()
#     val_df.head()
