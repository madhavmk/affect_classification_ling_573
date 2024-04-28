import json
import os
import re
import pandas as pd


from .data_processor import DataProcessor
from .feature_engineering import FeatureEngineering
from .classification_model import ClassificationModel
from .evaluation import Evaluator



def load_config(config_path):
    """
    Load config from a given path.
    
    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The loaded config if successful, otherwise None.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def make_eval_files(df, language):
    """
    Transform dataframes into desired formatted .tsv files for evaluation script. Includes the tweet ID, cleaned text,
    and either the gold standard data on HS, TR, and AG, or their predicted values from the model.

    Args:
        df (pandas dataframe): a dataframe containing both the gold standard and prediction classifications
        language (str): a string referring to the language of the dataframe content

    Returns:
        None
    """
    # check which language the data is in
    if language == "english":
        lang = "en"
    elif language == "spanish":
        lang = "es"

    # TASK A
    # split dataframe into gold and prediction dataframes
    gold_df = df[["cleaned_text", "HS", "TR", "AG"]].copy()
    pred_a_df = df[["HS_prediction"]].copy()

    # establish file paths, save dataframes as .tsv files
    # goldpath = "".join(["results/input/ref/", lang, ".tsv"])
    # predpath_a = "".join(["results/input/res/", lang, "_a.tsv"])
    goldpath = "".join(["outputs/D2/ref/", lang, ".tsv"])
    predpath_a = "".join(["outputs/D2/res/", lang, "_a.tsv"])
    gold_df.to_csv(goldpath, sep="\t") 
    pred_a_df.to_csv(predpath_a, sep="\t")


    # TASK B
    # split dataframe into gold and prediction dataframes
    pred_b_df = df[["HS_prediction", "TR_prediction", "AG_prediction"]].copy()

    pred_b_df = pred_b_df.rename(columns={"HS_prediction": "HS", "TR_prediction": "TR", "AG_prediction": "AG"})
    # establish file path, save dataframe as .tsv files
    # predpath_b = "".join(["results/input/res/", lang, "_b.tsv"])
    predpath_b = "".join(["outputs/D2/res/", lang, "_b.tsv"])
    pred_b_df.to_csv(predpath_b, sep="\t")





def main(config):
    """
    Run the system end-to-end using the config file

    Args:
        config (str): the path to a config.json file specifying the configuration of our system

    Returns:
        None
    
    """
    doc_config = config['document_processing']
    input_tsv_files = doc_config['input_tsv_files']
  
    # Initialize the class
    myDP = DataProcessor()
    # Load data from disk
    myDP.load_data(language= input_tsv_files['language'],
                   filepath = input_tsv_files['filepath']) 
                #    train_file = input_tsv_files['training'], 
                #    validation_file = input_tsv_files['devtest'])  

    # Clean the text
    myDP.clean_data()
  
    # Instantiate the FeatureEngineering object
    myFE = FeatureEngineering()

    # Fit
    train_df = myFE.fit_transform(myDP.processed_data['train'], 
                                embedding_file_path= config['model']['feature_engineering']['embedding_path'],
                                embedding_dim=25)
    
    # Transform
    val_df = myFE.transform(myDP.processed_data['validation'])

    # Instantiate the model
    myClassifier = ClassificationModel(config['model']['classification']['approach'])
    

    # Train the model
    features = ['percent_capitals', '!_count_normalized', '?_count_normalized', '$_count_normalized',
                '*_count_normalized', 'negative', 'positive', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
                'sadness', 'surprise', 'trust']
    embedding_features = ['Aggregate_embeddings']

    train_pred = myClassifier.fit(train_df,
                                tasks=['hate_speech_detection', 'target_or_general', 'aggression_detection'],
                                keep_training_data=False,
                                features=features,
                                embedding_features=embedding_features)


    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # # Save the results in the outputs directory
    # train_res = train_pred[['HS_prediction', 'TR_prediction', 'AG_prediction']]
    # train_res.to_csv('outputs/D2/train_results.csv')
    # val_res = val_pred[['HS_prediction', 'TR_prediction', 'AG_prediction']]
    # val_res.to_csv('outputs/D2/validation_results.csv')


    # create evaluation files based on val_pred
    make_eval_files(val_pred, input_tsv_files['language'])

    # Instantiate the evaluator and run it
    # myEvaluator = Evaluator("results/input", "results/output", config['evaluation']['output_file'])
    myEvaluator = Evaluator("outputs/D2", "outputs/D2", config['evaluation']['output_file'])

    myEvaluator.main()
    
       


if __name__ == "__main__":
    # config = load_config(config_path= os.path.join('..', 'config.json'))
    config = load_config(config_path='config.json')
    main(config)
