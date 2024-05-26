# Affect Classification Project 
## LING 573
## Teammates:
* [Ben Cote](https://github.com/bpcot23)
* [Madhav Kashyap](https://github.com/madhavmk)
* [Lindsay Skinner](https://github.com/skinnel)
* [Keaton Strawn](https://github.com/keatonstrawn)
* [Allan Tsai](https://github.com/chooshiba )

## Setup

To set up the project environment, follow the steps below:

1. Navigate to the "setup" folder using the command line:

   ```bash
   cd setup
   ```
2. Change the permission of the create_env.sh script to make it executable:
   
   ```bash
   chmod +x create_env.sh
   ```
4. Run the create_env.sh script to create the conda environment (you may need to modify the path depending on whether you are using anaconda or miniconda):
   
   ```bash
   ./create_env.sh miniconda_or_anaconda_directory_path_goes_here
   ```
6. Activate the newly created environment:
   
   ```bash
   conda activate Affect
   ```
Due to the size of the GloVe pretrained embeddings, we cannot store them in our GitHub repository. The user must separately download this file and place it in the specified directory.
To provide the model with GloVe embedding data, follow the steps below:
1. Go to the following link: https://zenodo.org/records/3237458 (please note that there are many sites where this file can be found. If you are unconfortable with the one provided, you should be able to find an alternative with a quick google search)
2. Here, scroll down and download the .gz file labeled "glove.twitter.27B.25d.txt.gz"
3. Once downloaded, unzip this file
4. Add the file "glove.twitter.27B.25d.txt" to the subdirectory `./data/`

Similarly, the size of the pickled dataframes are also too large to store on the GitHub repository. Please email to request them (or if you're an instructor, we will send them to you for the deliverable). Please add the pickled files to the subdirectory `./data/processed_data/`

## Components

- Data Processing

  - File: `src/data_processor.py`
  - This file contains functions and code for preprocessing raw data, including cleaning, formatting, and transforming the data. It also captures some raw text-based features.

- Feature Engineering

  - File: `src/feature_engineering.py`
  - This file contains methods for generating features for each tweet, including emotional lexicon lookups, embeddings, and normalization.

- nrc_lex_extension

  - File: `src/nrc_lex_extension.py`
  - This file contains the methods for applying emotion scores for each word in an embedding model, therefore extending the range of the nrc_lex feature.
     
- Classification

  - File: `src/classification_model.py`
  - This file contains methods for classifying each tweet as containing or not containing hate speech, being directed at a personal target or a group, and being or not being aggressive.

- Evaluation

  - File: `src/evaluation.py`
  - This file contains methods for evaluating model performance using precision, recall, accuracy, and F1 score.

   
## Running the System

To run the system from Condor:

1.   Activate the virtual environment and correctly download and place the GloVe embeddings file (see "Setup" above)
2.   Navigate to the scripts directory:
   ```bash
   cd ./scripts
   ```
3.    Change the permission of the run_main.sh script to make it executable:
   ```bash
   chmod +x run_main.sh
   ```
4.   Navigate to the main project directory:
   ```bash
   cd ..
   ```
5.   Submit the command file to Condor:
   ```bash
   condor_submit D4.cmd
   ```
6.   Check system progress:
   ```bash
   condor_q
   ```
7.   To see system performance evaluation, use the following command to navigate to the system scores directory from the home directory:
   ```bash
   cd ./D4
   ```
8.   To see the system's prediction files, use the following command to navigate to the evaluation data directory from the home directory:
   ```bash
   cd ./outputs/D4/res
   ```
9.   To see the system's score file, use the following command to navigate to the results directory from the home directory:
10.   ```bash
      cd ./results
      ```

## Scripts
- `scripts/run_main.sh`: This is a script that runs the system per the parameters found in `config.json` Usage:
   ```bash
  #!/bin/sh
   if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
       source "$HOME/anaconda3/etc/profile.d/conda.sh"
       conda activate Affect
   elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
       source "$HOME/miniconda3/etc/profile.d/conda.sh"
       conda activate Affect
   fi
   
   python3  src/main.py
   ```
Running this shell script on its own will result in errors because the file paths defined in the code are based on the location of the D3.cmd file for remote running on Condor.
Further, if the user's virtual environments are contained in paths other than one of the two represented above, they will need to modify the script to match their path.
  
  
## Configs
All arguments for the system are passed through the config file (`config.json`):

- `"document_processing"`: identifies arguments associated with ingesting and processing the original training and development .tsv files.
   - `"input_tsv_files"`: identifies the arguments associated with the file paths and language of the model.
      - `"filepath"`: argument identifying the directory within the repository that contains all the data.
      - `"training"`: argument specifying the file path to the training data.
      - `"devtest"`: argument specifying the file path to the development/testing data.
      - `"language"`: argument specifying the language of the data being used.
   - `"save_or_load"`: argument specifying whether the model will be run with pretrained loaded data or not.
   - `"processed_data_dir"`: argument specifying the file path to the pretrained data.
- `"model"`: identifies arguments associated with the core affect classification modules of the system.
   - `"feature_engineering"`: identifies arguments associated with the development of features for the model.
      - `"embedding_path"`: argument specifying the file path where the GloVe embedding data should be stored.
      - `"embedding_dim"`: argument specifying the dimension number for the GloVe embeddings generated by the model.
      - `"slang_dict_path"`: argument specifying the file path where the slang dictionary is stored.
      - `"span_NRC_path"`: argument specifying the file path where the Spanish NRC dictionary is stored.
      - `"load_translations"`: argument specifying whether to load or save Spanish translations during training.
      - `"trans_path"`: argument specifying the file path to where the saved Spanish translations are stored, if present.
      - `"stop_words_path"`: argument specifying the file path where the stop words list is stored.
      - `"es_sent_path"`: argument specifying the file path where the spanish word sentiment dictionary is stored.
      - `"nrc_embedding_file"`: argument specifying the file path where the GloVe embedding file is, for the nrclex extension module.
   - `"classification"`: identifies arguments associated with the classification of model data.
      - `"approach"`: argument specifying the classification approached used by the model.
      - `"params"`: identifies argument parameters associated with the given classification approach.
         - `"tasks"`: argument list specifying the tasks for which the model will try to predict data classification.
         - `"keep_training_data"`: boolean argument. If the value is set to `true`, the system will save the training data from the model. If the value is set to `false`, the system will not save training data.
         - `"prediction_target"`: argument specifying whether the classification is performed on three binary metrics (HS 0/1, TR 0/1, AG 0/1) or a five-way classification (HS+TR+AG, HS+TR, HS+AG, HS, None).
         - `"features"`: argument list specifying the header labels in the dataframe that represent the model's non-embedding-based features.
         - `"embedding_features"`: argument list specifying the header labels in the dataframe that represent the model's embedding-based features.
         - `"model_params"`: identifies the parameter sets used for training each classifier model
            - `"random_forest_params"`: identifies argument parameters associated with the training of the random forest classifier
            - `"svm_params"`: identifies argument parameters associated with the training of the svm classifier  
- `"evaluation"`: identifies arguments associated with evaluation of the system's performance.
   - `"output_file"`: argument specifying the name of the file containing the model's performance scores.
   - `"input_directory"`: argument specifying the input directory for the data used to generate evaluations.
   - `"output_directory"`: argument specifying the output directory for the output files of the evaluation score results.
   - `"goldpath"`: argument specifying the path for where the ground truth data is stored for the evaluator model.
   - `"predpath"`: argument specifying the path for where the model's predicted classification data is stored for the evaluator model. 


## (Optional) Running RoBERTa model
Download our trained and finetuned RoBERTa model weights for English Task B from https://drive.google.com/file/d/1GG1wyZtNhC226WWb25DOcfjV_Y0wrvHc/view?usp=sharing

Train using the jupyter notebook `src/training_roberta_model.ipynb`
