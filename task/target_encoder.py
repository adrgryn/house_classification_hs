import data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import category_encoders as ce


df = data.create_df()

# Create X and y datasets
X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values,
                                                    random_state=1)

# TargetEncoder
enc = ce.TargetEncoder(cols=['Zip_area', 'Zip_loc', 'Room'])
enc.fit(X_train, y_train)

X_train_encoded = enc.transform(X_train)
X_test_encoded = enc.transform(X_test)


# Creating a DecisionTreeClassifier object
clf = DecisionTreeClassifier(criterion="entropy", max_features=3, splitter='best', max_depth=6,
                             min_samples_split=4, random_state=3)

# Fit the model to the training data
clf.fit(X_train_encoded, y_train)

y_pred = clf.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
