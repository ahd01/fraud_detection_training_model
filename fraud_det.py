import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import mlflow

credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.head()
credit_card_data.tail()
credit_card_data.info()

#distribution of legit transaction and fruadlent transaction 
credit_card_data['Class'].value_counts()

#sperating data for analysis 
legit = credit_card_data[credit_card_data.Class == 0]
fruad = credit_card_data[credit_card_data.Class == 1]

#statistical measure of the data 
legit.Amount.describe()
fruad.Amount.describe()

#compare the values for both transactions
credit_card_data.groupby('Class').mean()

print(legit.shape)
print(fruad.shape)

#build a sample dataset similar distributions of normal transaction and fruadlent transactions
legit_sample = legit.sample(n=492)

#concat two data frames 
new_dataset = pd.concat([legit_sample,fruad], axis=0)
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

#samplitting data into features and targets 
X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 2)
# Scale the training data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
mlflow.log_param("C", model.C)  # Regularization parameter
# Make predictions on the training and test data
X_train_scaled_without_names = X_train_scaled.copy()
X_train_scaled_without_names = X_train_scaled_without_names.values if isinstance(X_train_scaled_without_names, pd.DataFrame) else X_train_scaled_without_names
X_train_prediction = model.predict(X_train_scaled_without_names)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training data accuracy is:', training_data_accuracy)
X_test_scaled = scaler.transform(X_test)
X_test_prediction = model.predict(X_test_scaled)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Test data accuracy is:', test_data_accuracy)
mlflow.log_metric("training_accuracy", training_data_accuracy)
mlflow.log_metric("test_accuracy", test_data_accuracy)
mlflow.sklearn.log_model(model, "logistic_regression_model")
#Assuming the code for model training and evaluation has been run
training_data_accuracy = 0.85  # Example training accuracy
test_data_accuracy = 0.78  # Example test accuracy

# Data for plotting
labels = ['Training Data', 'Test Data']
accuracy_scores = [training_data_accuracy, test_data_accuracy]

# Creating the bar chart
plt.bar(labels, accuracy_scores, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set the y-axis limits
plt.title('Model Accuracy on Training and Test Data')
plt.show()