import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer

dataset = pd.read_csv('life_expectancy.csv')

# First 5 entries in the dataset
#print(dataset.head(5))
# The last column - that being life expectancy
#labels = dataset.iloc[:,-1]
#print(labels.describe())

# Removing the country column using drop method
dataset = dataset.drop("Country", axis="columns")

# Splitting data into labels and functions
labels = dataset.iloc[:,-1] 
features = dataset.iloc[:, :-1]
#print(features)


# |--------Data Preprocessing----------|


# Turning categorical columns into numerical columns
features = pd.get_dummies(features)

# Splitting data into training set and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=23)

# Standardising/normalising your numerical features
num_features = features.select_dtypes(include=['int64', 'float64']).columns
ct = ColumnTransformer([('normalize', Normalizer(), num_features)], remainder = 'passthrough')

# Fitting instance ct to the training data and transforming it
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)


# |--------Building the model----------|





