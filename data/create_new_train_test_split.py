""" This script takes in the original training and validation datasets and creates a new train-validation-test split of
the data, which are saved as .tsv files"""

# Libraries
import pandas as pd
from copy import deepcopy

# Load the original data from disk
# English
raw_train_en_df = pd.read_csv(f'data/original_data/train_en.tsv', sep='\t', header=0, index_col='id')
raw_val_en_df = pd.read_csv(f'data/original_data/dev_en.tsv', sep='\t', header=0, index_col='id')
# Spanish
raw_train_es_df = pd.read_csv(f'data/original_data/train_es.tsv', sep='\t', header=0, index_col='id')
raw_val_es_df = pd.read_csv(f'data/original_data/dev_es.tsv', sep='\t', header=0, index_col='id')

# Create a new test dataset using a random 20% of the training data
# English
test_en_df = deepcopy(raw_train_en_df.sample(frac=0.2, replace=False))
# Spanish
test_es_df = deepcopy(raw_train_es_df.sample(frac=0.2, replace=False))

# Remove that data from the training data
# English
train_en_df = deepcopy(raw_train_en_df)
train_en_df = train_en_df.drop(test_en_df.index)
# Spanish
train_es_df = deepcopy(raw_train_es_df)
train_es_df = train_es_df.drop(test_es_df.index)

# Save results as .tsv files in the data directory
# English
train_en_df.to_csv('data/train_en.tsv', sep='\t')
raw_val_en_df.to_csv('data/dev_en.tsv', sep='\t')
test_en_df.to_csv('data/test_en.tsv', sep='\t')
# Spanish
train_es_df.to_csv('data/train_es.tsv', sep='\t')
raw_val_es_df.to_csv('data/dev_es.tsv', sep='\t')
test_es_df.to_csv('data/test_es.tsv', sep='\t')
