# Comparative-Analysis-of-Machine-Learning-Approaches-to-Detect-Malicious-URLs-
A Comparative Analysis of Machine Learning Approaches to Detect Malicious URLs for Phishing Attack Prevention

## Abstract

Malicious URL attacks, often use duplicate websites and authentic logos to deceive users, which pose a significant threat in cyberspace, leading to the disclosure of sensitive information like bank details and passwords. Despite existing anti-phishing measures, attackers continuously evolve their tactics,highlighting the need for an effective prediction mechanism to protect users effectively. Classification is one of the techniques used to detect website phishing. This paper has proposed a model for detecting phishing attacks using various machine learning (ML) classifiers. Random Forest, CATB, and Naïve Bayes are used as the machine learning classifiers to train the proposed model.The dataset in this research was obtained from the public online repository Mendeley with 48 features are extracted from 5000 phishing websites and 5000 real websites. The model was analyzed using F1 scores, where both precision and recall evaluations are taken into consideration. The proposed work has concluded that the random forest classifier has achieved the most efficient and highest performance scoring with 98% accuracy.
## Methodology
-Overview:
Our research primarily utilizes machine learning techniques to train models capable of detecting phishing websites. We analyze data to find patterns and relationships, then test the model and measure its performance. Based on the model's performance,the training procedure and dataset preprocessing are adjusted to improve results in subsequent iterations. If the model performs better than all others, it is stored and used on new unknown datasets to further verify its performance.

- Data Collection – Dataset:
We gathered a dataset of 10,000 websites (5,000 phishing and 5,000 legitimate) from a public online repository. The dataset contains 48 features extracted from each website. We used this data to train our models to detect phishing websites.

- Data Preprocessing – Exploratory Data Analysis (EDA):
Before training our models, we needed to clean and prepare the data. We checked for missing values and duplicates, and removed any garbage values that could affect our models' performance. We also standardized the features to ensure consistency and selected the most important features to use in our models[8]. Additionally, we handled outliers to prevent them from affecting our models' performance. This preprocessing step is essential to ensure that our models are trained on high-quality data.

- Cross-Validation – Splitting the dataset into training and test set:
To evaluate our models' performance, we split the dataset into two parts: a training set (80% of the data) and a test set (20% of the data). We used the training set to train our models and the test set to evaluate their performance. This process, known as cross-validation, helps us to avoid overfitting and ensure that our models generalize well to new data.

- Model_Selection & Model_Fitting:
We selected three machine learning models: Random Forest, CatBoost, and Naive Bayes. We trained each model on the training set using the selected features and fine-tuned their hyperparameters to optimize their performance. This step is critical in identifying the best model for our task.

- Testing Models:
We tested each model on the test set to evaluate their performance. We generated predictions for each model and compared them to the actual labels. We calculated metrics such as accuracy, precision, and recall to evaluate each model's performance. This step helps us to identify the strengths and weaknesses of each model.

- Evaluation of Models:
We evaluated each model's performance using metrics such as accuracy, precision,recall, and F1-score[1-3,7,10]. We generated confusion matrices to visualize each model's performance and identified the strengths and weaknesses of each model. This step is essential in identifying the best-performing model.

- Model Comparison & Saving:
We compared the performance of each model and identified the best-performing model. We saved the best-performing model using joblib for future use. We also documented the results and insights gained from the model comparison. This step marks the completion of our project, and we can now use the best-performing model to detect phishing website
