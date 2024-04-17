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
2. Change the permission of the create_env.sh script to make it executable"
   
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
   
## Running the System

To run the system from a python console activate the virtual environment (see above) and complete the following steps. 
Alternatively, users can run the main.py script, which executes the workflow described below.

1. Import all required modules.

   ```python
   from src.data_processor import DataProcessor
   from src.feature_engineering import FeatureEngineering
   from src.classification_model import ClassificationModel
   # import ModelEvaluator class
   
   from copy import deepcopy
   ```

2. Load and clean the raw data.

   ```python
    # Instantiate the DataProcessor object
    myDP = DataProcessor()

    # Load data from disk
    myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac

    # Clean the text
    myDP.clean_data()
   ```

3. Generate features for the model to use.

   ```python
    # Instantiate the FeatureEngineering object
    myFE = FeatureEngineering()

    # Fit the feature generators
    train_df = myFE.fit_transform(myDP.processed_data['train'], embedding_file_path='../data/glove.twitter.27B.25d.txt',
        embedding_dim=25)

    # Transform the validation data
    val_df = myFE.transform(myDP.processed_data['validation'])
   ```

4. Train the baseline model.

   ```python
    # Instantiate the model
    myBaseline = ClassificationModel('baseline')

    # Train the model
    train_df_baseline = deepcopy(train_df)
    train_pred_baseline = myBaseline.fit(train_df_baseline, tasks=['hate_speech_detection'], keep_training_data=False)

    # Run the model on the validation data
    val_df_baseline = deepcopy(val_df)
    val_pred_baseline = myBaseline.predict(val_df_baseline)
   ```

5. Train the classification model.

   ```python
    # Instantiate the model
    myClassifier = ClassificationModel('random_forest')

    # Train the model
    # TODO: features list should be powered by the config file
    features = ['percent_capitals', '!_count_normalized', '?_count_normalized', '$_count_normalized', 
                '*_count_normalized', 'negative', 'positive', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 
                'sadness', 'surprise', 'trust']
    train_pred_rf = myClassifier.fit(train_df, tasks=['hate_speech_detection'], keep_training_data=False, 
                                     features=features, embedding_features=['Aggregate_embeddings'])

    # Run the model on the validation data
    val_pred_rf = myClassifier.predict(val_df)
   ```

6. Evaluate the model's performance.

   ```python
   # Put ModelEvaluator-affiliated code here
   ```