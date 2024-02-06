# house_classification_hs

This project aims to predict house prices based on various features using a DecisionTreeClassifier. The model's performance is evaluated using three different encoding techniques: OneHotEncoder, OrdinalEncoder, and TargetEncoder.

Project Structure
The project is divided into several scripts, each responsible for a different part of the machine learning pipeline:
data.py: Contains the logic to download and load the dataset.
one_hot_encode.py, ordinal_encoder.py, target_encoder.py: Each script applies a different encoding technique to the categorical features and evaluates the model's performance.
solution.py: Compare F1-score macro average values

Installation
Ensure you have Python installed on your system. Then, install the necessary libraries using:
pip install pandas, scikit-learn, requests, category_encoders

Usage
Run the main script to evaluate the model's performance with different encoders:
python solution.py


Dataset
The dataset house_class.csv includes features such as Area, Room, Longitude (Lon), Latitude (Lat), Zip_area, and Zip_loc, with the target variable being Price.

Model Evaluation
The models are evaluated based on precision, recall, and F1-score metrics for each encoding technique. The main script prints out the F1-score macro average value for comparison:
OneHotEncoder: F1-score
OrdinalEncoder: F1-score
TargetEncoder: F1-score

Encoding Techniques
OneHotEncoder: Converts categorical variables into a form that could be provided to ML algorithms to do a better prediction.
OrdinalEncoder: Encodes categorical features as an integer array, considering the order of the categories.
TargetEncoder: Uses the means of the target variable for each category to encode the features.

Model
A DecisionTreeClassifier with the following parameters is used for prediction:
criterion: "entropy"
max_features: 3
splitter: 'best'
max_depth: 6
min_samples_split: 4
random_state: 3

Conclusion
This project demonstrates how different encoding techniques can affect the performance of a machine learning model. By comparing the F1-score macro average values, one can select the most suitable encoding technique for this dataset.
