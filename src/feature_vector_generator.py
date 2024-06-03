""" This script defines a function to take in the dataframe that results from the FeatureEngineering transform method
and generate a dataframe of feature-vectors, where each cell contains a single value rather than e.g. full embeddings.
"""

# Libraries
import pandas as pd
import numpy as np
import pickle as pkl
from typing import List, Optional

# Define helper function
def target_processing(data: pd.DataFrame, tasks: Optional[List[str]] = ['HS', 'TR', 'AG']):
    """Processes the target categories into a uniform format. So rather than having e.g. 3 binary categories for HS,
    TR and AG (with dependencies) we have a single 5 category problem (HS, HS+TR, HS+AG, HS+TR+AG, NotHS).

    Arguments:
    ----------
    data
        The dataset, with the target task columns.
    tasks
        The list of tasks being targeted. Can be any non-empty combination of 'HS', 'TR', and 'AG'. Default behavior is
        to include all tasks.


    Returns:
    --------
    A dataframe containing the new target category with the same index as the original dataframe.
    """

    # Specify which columns contain the target class(es) and order them
    task_cols = []
    if 'HS' in tasks:
        task_cols.append('HS')
    if 'TR' in tasks:
        task_cols.append('TR')
    if 'AG' in tasks:
        task_cols.append('AG')

    # Specify classification objective
    target_categories = []
    for index, row in data.iterrows():
        targets = row[task_cols]
        # Specify if class is HS or not
        if targets.iloc[0] == 1:
            new_category = f'{task_cols[0]}+'
            # If class is HS, is it TR?
            if targets.iloc[1] == 1:
                new_category = f'{new_category}{task_cols[1]}+'
            # If class is HS, is it AG?
            if targets.iloc[2] == 1:
                new_category = f'{new_category}{task_cols[2]}'
            # Remove trailing +, if it exists
            new_category = new_category.rstrip('+')
        else:
            new_category = 'NotHS'
        target_categories.append(new_category)
    target_df = pd.DataFrame(target_categories, index=data.index)

    return target_df

# Define main function
def feature_vector_generator(data: pd.DataFrame,
                             features: Optional[List[str]] = 'All',
                             embedding_features: Optional[List[str]] = None,
                             tasks: Optional[List[str]] = ['HS', 'TR', 'AG']) -> pd.DataFrame:
    """Generates a dataframe that treats each column as an individual (single value) feature, resulting in rows of
    feature vectors.

    Arguments:
    ----------
    data
        DataFrame resulting from FeatureEngineering that contains all of the relevant features used to generate the
        feature vector.
    features
        The list of non-embedding features to be included in the feature vector. Default behavior is to include all
        available features.
    embedding_features
        The list of embedding features to be included in the feature vector. Each dimension of an embedding will appear
        as a single column in the feature vector.
    tasks
        The list of tasks being targetted. May be any non-empty combination of 'HS', 'TR', and 'AG'. Default behavior is
        to include all tasks.


    Returns:
    --------
    feature_vecs
        DataFrame whose rows correspond to feature vectors. Original index from data object is maintained.
    task_vals
        DataFrame whose rows correspond to the binary task labels. Original index from data object is maintained.
    target_vals
        DataFrame whose rows correspond to the multi-class classification task labels (e.g. 'NotHS', 'HS', 'HS+TR',
        etc.) Original index from data object is maintained.
    """

    # Process features
    if features == 'All':
        features = ["percent_capitals", "!_count_normalized", "?_count_normalized", "$_count_normalized",
                    "*_count_normalized", "negative", "positive", "anger", "anticipation", "disgust", "fear", "joy",
                    "sadness", "surprise", "trust", "negative_ext", "positive_ext", "anger_ext", "anticipation_ext",
                    "disgust_ext", "fear_ext", "joy_ext", "sadness_ext", "surprise_ext", "trust_ext", "slangscore"]

    # Get task values
    task_vals = data[tasks]

    # Get target values
    target_vals = target_processing(data, tasks)

    # Limit training data to include only the features
    feat_vecs = data[features]

    # Process embedding(s) features, if they exist
    if embedding_features is not None:
        for ef in embedding_features:
            embeddings = np.stack(data[ef])
            col_prefix = f'{ef}_dim_'
            emb_cols = [col_prefix + str(dim) for dim in range(embeddings.shape[1])]
            embeddings = pd.DataFrame(embeddings, columns=emb_cols, index=data.index)
            feat_vecs = pd.concat([feat_vecs, embeddings], axis=1)

    return feat_vecs, task_vals, target_vals


if __name__ == "__main__":
    # Load data for testing
    data_file = '../data/processed_data/D4/original_data/train_fe_df.pkl'
    with open(data_file, 'rb') as f:
        data = pkl.load(f)

    # Specify features to be included
    features = ["percent_capitals", "!_count_normalized", "?_count_normalized", "$_count_normalized", "*_count_normalized",
                "negative_ext", "positive_ext", "anger_ext", "anticipation_ext", "disgust_ext", "fear_ext", "joy_ext",
                "sadness_ext", "surprise_ext", "trust_ext", "slangscore"]

    emb_features = ['Universal_Sentence_Encoder_embeddings', 'Aggregate_embeddings', 'BERTweet_embeddings']


    feature_vectors, task_values, target_values = feature_vector_generator(data=data,
                                                                           features=features,
                                                                           embedding_features=emb_features,
                                                                           tasks=['HS', 'TR', 'AG']
                                                                           )

    df = pd.concat([feature_vectors, target_values], axis=1)

    class_counts = df[0].value_counts()
    max_count = class_counts.max()

    print(class_counts)

    np.random.seed(611)
    new_instances = []
    for label in class_counts.index:
        class_df = df[df[0] == label]
        current_count = class_counts[label]
        while current_count < max_count:
            sample = class_df.sample(n=1, replace=True)
            noise = np.random.normal(0, 0.01, sample.shape[1] - 1)
            new_sample = sample.iloc[:, :-1].values + noise
            new_instance = pd.DataFrame(new_sample, columns=sample.columns[:-1])
            new_instance[0] = label
            new_instances.append(new_instance)
            current_count += 1

    df_new = pd.concat(new_instances, ignore_index=True)
    df_balanced = pd.concat([df, df_new], ignore_index=True)

    # shuffle - if this matters
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    print("new class dist:")
    print(df_balanced[0].value_counts())

