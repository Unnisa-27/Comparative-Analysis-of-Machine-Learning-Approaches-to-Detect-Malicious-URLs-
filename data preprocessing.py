#Data Pre-processing

# To identify column that doesnt have any significance
df.describe()
# Display correlation martix in heatmap
import seaborn as sns
plt.figure(figsize=(45,45))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# 1. checking the data for null or missing values
df.isnull().sum()
# 2. Finding Duplicate Values
df.duplicated().sum()
# 3. Finding Gargabe Values
for i in df.select_dtypes(include='object').columns:
    print(df[i].value_counts())
    print("*"*10)
# 4. Boxplot-to-identify Outliers
import warnings
warnings.filterwarnings("ignore")
for i in df.select_dtypes(include='number').columns:
    sns.boxplot(data=df,x=i)
    plt.show()

# Removes all outliers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to detect and treat outliers using the IQR-based method
def detect_and_treat_outliers_iqr(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column_name] < lower_limit) | (df[column_name] > upper_limit)]

    # Trim outliers
    df_no_outliers = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]

    # Cap outliers
    df_cap = df.copy()
    df_cap[column_name] = np.where(df_cap[column_name] > upper_limit, upper_limit, df_cap[column_name])
    df_cap[column_name] = np.where(df_cap[column_name] < lower_limit, lower_limit, df_cap[column_name])

    return df_no_outliers, df_cap, outliers

# Apply the function to each column in the DataFrame
for column_name in df.columns:
    if df[column_name].dtype != 'object':
        print(f"Processing column: {column_name}")
        df_no_outliers, df_cap, outliers = detect_and_treat_outliers_iqr(df, column_name)

        # Visualize the outliers before and after treatment
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df[column_name])
        plt.title('Before Treatment')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df_cap[column_name])
        plt.title('After Treatment')
        plt.show()

        # Update the DataFrame with the treated data
        df = df_cap
      
# Splitting the data into features and target
target = df1['CLASS_LABEL']
features = df1.drop(['id', 'CLASS_LABEL'], axis=1)

from sklearn.preprocessing import StandardScaler

# Preprocessing: Scale features to have mean=0 and std=1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
