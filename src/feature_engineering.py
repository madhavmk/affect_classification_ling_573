"""This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import pandas as pd
from nrclex import NRCLex
from typing import Optional
from nltk.tokenize import word_tokenize

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

    def _NRC_counts(self, data: pd.DataFrame):
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

    def fit_transform(self, train_data):
        """Learns all necessary information from the provided training data in order to generate the complete set of
        features to be fed into the classification model. In the fitting process, the training data is also transformed
        into the feature-set expected by the model and returned.

        Arguments:
        ---------
        train_data
            The training data that is used to define the feature-engineering methods.

        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the calculated features for each observation
            in the dataset.
        """

        # Get the training data, to be used for fitting
        self.train_data = train_data

        # Framework to add in steps for each feature that is to be generated
        # transformed_data = self._example_feature1_method(train_data, fit=True, other_args=None)
        transformed_data = self._NRC_counts(train_data)

        # TODO: add in code below to fit and transform training data to generate other features as they are added

        # Update the fitted flag
        self.fitted = True

        return transformed_data

    def transform(self, data):
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

        # Framework to add in steps for each feature that is to be generated
        transformed_data = self._NRC_counts(data)

        # TODO: add in code below to transform datasets to generate other features as they are added

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
    train_df = myFE.fit_transform(myDP.processed_data['train'])

    # Transform
    val_df = myFE.transform(myDP.processed_data['validation'])

    # View a sample of the results
    train_df.head()
    val_df.head()


