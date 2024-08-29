#Model_Selection & Fitting
!pip install catboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Instantiate the models
rf_model = RandomForestClassifier()

# Define the number of features you want to select
num_features_to_select = 10  # You can adjust this as needed

# Initialize RFE for each model
rf_rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=num_features_to_select)

# Fit RFE to the training data and transform features
X_train_rf_rfe = rf_rfe.fit_transform(X_train, y_train)

# Transform test data using the selected features
X_test_rf_rfe = rf_rfe.transform(X_test)

# Fit models to the selected features
rf_model_rfe = RandomForestClassifier()

rf_model_rfe.fit(X_train_rf_rfe, y_train)


# Make predictions on the test data
rf_y_pred_rfe = rf_model_rfe.predict(X_test_rf_rfe)

from catboost import CatBoostClassifier

# Instantiate the models
cat_model = CatBoostClassifier()


# Define the number of features you want to select
num_features_to_select = 10  # You can adjust this as needed

# Initialize RFE for each model
cat_rfe = RFE(estimator=CatBoostClassifier(), n_features_to_select=num_features_to_select)

# Fit RFE to the training data and transform features
X_train_cat_rfe = cat_rfe.fit_transform(X_train, y_train)

# Transform test data using the selected features
X_test_cat_rfe = cat_rfe.transform(X_test)

# Fit models to the selected features
cat_model_rfe = CatBoostClassifier()

cat_model_rfe.fit(X_train_cat_rfe, y_train)

# Make predictions on the test data
cat_y_pred_rfe = cat_model_rfe.predict(X_test_cat_rfe)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Naive Bayes classifier expects non-negative input, so we'll transform the data
# Convert negative values to positive using absolute value function
X_train_nb = np.abs(X_train)
X_test_nb = np.abs(X_test)
mul_nb_model = MultinomialNB()
gaussian_nb_model = GaussianNB()
bernoulli_nb_model = BernoulliNB()
complement_nb_model = ComplementNB()
categorical_nb_model = CategoricalNB()

# Define the number of features you want to select
num_features_to_select = 10  # You can adjust this as needed

# Initialize RFE for each model
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=num_features_to_select)

# Fit RFE to the training data and transform features
X_train_nb = rfe.fit_transform(X_train_nb, y_train)

# Transform test data using the selected features
X_test_nb = rfe.transform(X_test_nb)


# Fit the models to the training data
mul_nb_model.fit(X_train_nb, y_train)
gaussian_nb_model.fit(X_train_nb, y_train)
bernoulli_nb_model.fit(X_train_nb, y_train)
complement_nb_model.fit(X_train_nb, y_train)
categorical_nb_model.fit(X_train_nb, y_train)

# Make predictions on the test data
mul_nb_y_pred = mul_nb_model.predict(X_test_nb)
gaussian_nb_y_pred = gaussian_nb_model.predict(X_test_nb)
bernoulli_nb_y_pred = bernoulli_nb_model.predict(X_test_nb)
complement_nb_y_pred = complement_nb_model.predict(X_test_nb)
categorical_nb_y_pred = categorical_nb_model.predict(X_test_nb)

#Accuracy
mul_nb_acc = accuracy_score(y_test, mul_nb_y_pred)
gaussian_nb_acc = accuracy_score(y_test, gaussian_nb_y_pred)
bernoulli_nb_acc = accuracy_score(y_test, bernoulli_nb_y_pred)
complement_nb_acc = accuracy_score(y_test, complement_nb_y_pred)
categorical_nb_acc = accuracy_score(y_test, categorical_nb_y_pred)

print("Multinomial Naive Bayes Accuracy:", mul_nb_acc)
print("Gaussian Naive Bayes Accuracy:", gaussian_nb_acc)
print("Bernoulli Naive Bayes Accuracy:", bernoulli_nb_acc)
print("Complement Naive Bayes Accuracy:", complement_nb_acc)
print("Categorical Naive Bayes Accuracy:", categorical_nb_acc)

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB

# Naive Bayes classifier expects non-negative input, so we'll transform the data
# Convert negative values to positive using absolute value function
X_train_nb = np.abs(X_train)
X_test_nb = np.abs(X_test)
mul_nb_model = MultinomialNB()
gaussian_nb_model = GaussianNB()
bernoulli_nb_model = BernoulliNB()
complement_nb_model = ComplementNB()
categorical_nb_model = CategoricalNB()

# Define the number of features you want to select
num_features_to_select = 10  # You can adjust this as needed

# Initialize RFE for each model
rfe = RFE(estimator=CatBoostClassifier(), n_features_to_select=num_features_to_select)

# Fit RFE to the training data and transform features
X_train_nb = rfe.fit_transform(X_train_nb, y_train)

# Transform test data using the selected features
X_test_nb = rfe.transform(X_test_nb)


# Fit the models to the training data
mul_nb_model.fit(X_train_nb, y_train)
gaussian_nb_model.fit(X_train_nb, y_train)
bernoulli_nb_model.fit(X_train_nb, y_train)
complement_nb_model.fit(X_train_nb, y_train)
categorical_nb_model.fit(X_train_nb, y_train)

# Make predictions on the test data
mul_nb_y_pred = mul_nb_model.predict(X_test_nb)
gaussian_nb_y_pred = gaussian_nb_model.predict(X_test_nb)
bernoulli_nb_y_pred = bernoulli_nb_model.predict(X_test_nb)
complement_nb_y_pred = complement_nb_model.predict(X_test_nb)
categorical_nb_y_pred = categorical_nb_model.predict(X_test_nb)

#Accuracy
mul_nb_acc = accuracy_score(y_test, mul_nb_y_pred)
gaussian_nb_acc = accuracy_score(y_test, gaussian_nb_y_pred)
bernoulli_nb_acc = accuracy_score(y_test, bernoulli_nb_y_pred)
complement_nb_acc = accuracy_score(y_test, complement_nb_y_pred)
categorical_nb_acc = accuracy_score(y_test, categorical_nb_y_pred)

print("Multinomial Naive Bayes Accuracy:", mul_nb_acc)
print("Gaussian Naive Bayes Accuracy:", gaussian_nb_acc)
print("Bernoulli Naive Bayes Accuracy:", bernoulli_nb_acc)
print("Complement Naive Bayes Accuracy:", complement_nb_acc)
print("Categorical Naive Bayes Accuracy:", categorical_nb_acc)

from sklearn.metrics import accuracy_score
# Calculate accuracies
rf_acc = accuracy_score(y_test, rf_y_pred_rfe )
cat_acc = accuracy_score(y_test, cat_y_pred_rfe )
nb_acc = accuracy_score(y_test, categorical_nb_y_pred)

print("Random Forest Accuracy: ", rf_acc)
print("CatBoost Accuracy: ", cat_acc)
print("Naive Bayes Accuracy: ", nb_acc)

import matplotlib.pyplot as plt

# Create a list of models and their corresponding accuracy scores
models = ['Random Forest', 'CATB', 'Naive bayes']
train_accuracies = [rf_model_rfe.score(X_train_rf_rfe, y_train), cat_model_rfe.score(X_train_cat_rfe, y_train), categorical_nb_model.score(X_train_nb, y_train)]
test_accuracies = [rf_acc, cat_acc, nb_acc]


# Plot the training and testing accuracy for each model
plt.figure(figsize=(10, 6))
plt.bar(models, train_accuracies, color='blue', label='Training Accuracy')
plt.bar(models, test_accuracies, color='orange', label='Testing Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy for Each Model')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.legend()
plt.show()
