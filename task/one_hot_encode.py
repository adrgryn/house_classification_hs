import pandas as pd
import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


df = data.create_df()
# Create X and y datasets
X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values,
                                                    random_state=1)

# Create and configure encoder
enc = OneHotEncoder(drop='first')
enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

# Transform the training and the test datasets with the fitter encoder
X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                   index=X_train.index).add_prefix('enc')

X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                  index=X_test.index).add_prefix('enc')

# Return the transformed data to the dataset
X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

# Creating a DecisionTreeClassifier object
clf = DecisionTreeClassifier(criterion="entropy", max_features=3, splitter='best', max_depth=6,
                             min_samples_split=4, random_state=3)

# Fit the model to the training data
clf.fit(X_train_final, y_train)

y_pred = clf.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)

