File description:

build_features.py - This file is used to perform Feature Engineering and extract features from the dataset.

ml_modelling.py - Includes code for Data Cleaning, Training, Cross-validation and Testing.

three_points_extracted - It is a text file containing the X, Y and Z co-ordinates. This file will be used in the build_features.py file to find the three closest points, given a point.

Semantic Classificatin of Point Cloud using Machine Learning.pdf - It is a report containing the description of the whole project.

Due to size constraints, the datasets are not uploaded as a part of this git repository. In order to run the build_features.py/ml_modelling.py, you need to download the datasets from the link below: 

(https://exchangelabsgmu-my.sharepoint.com/:f:/g/personal/ckasula_masonlive_gmu_edu/EgCzw1QVCJFKtL8-_JAAtCUBwwXlZZaon49FSBj4k_R3og?e=rVwOpp) 

The description of the files in the above link is as below.

cleaned_dataset_500001.csv  - A cleaned pre-processed dataset of 500,000 records of the Vaihingen dataset.
dataset_500000.csv          - A dataset of 500,000 records from the Vaihingen dataset which consists of the hand-built features. This is a product of the build_features.py file.
random_forest_model.sav     - A trained random forest model saved into a .pkl file.
Vaihingen3D_Traininig.csv   - The raw dataset extracted directly from the source (https://www2.isprs.org/commissions/comm2/wg4/benchmark/3d-semantic-labeling/). 


Execution time:

After placing the datasets and the codes in a single folder, you can run the ml_modelling.py to train, test and save the random forest model. You dont have to run the build_features.py file, as its product is already provided as dataset_500000.csv. 

ml_modelling.py takes 30-45 mins to get executed.

build_features.py takes around 5 days to generate dataset_500000.csv when executed 24x7.
