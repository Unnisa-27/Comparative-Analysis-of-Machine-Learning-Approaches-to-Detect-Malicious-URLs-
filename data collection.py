#Data Collection

import pandas as pd
#Loading the data
df = pd.read_csv("/content/Phishing_Legitimate_full.csv")
df
df1=df.copy()
df1.head()
import numpy as np
import random
np.random.seed(42)
random.seed(42)
# Checking the shape of dataset
print("Shape:")
df.shape
# Listing features of dataset
print("Features:")
df.columns
# Information about dataset
print("Information: \n")
df.info()
# To know count of class_label
print("\nCount of Class_labels:")
class_counts = df['CLASS_LABEL'].value_counts()
print(class_counts)
import matplotlib.pyplot as plt
# Visualize class_labels
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Class Label Counts')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()
# Plotting the data distribution
df.hist(bins = 50,figsize = (20,20))
plt.show()
# Heatmap
import seaborn as sns
plt.figure(figsize=(15,15))
sns.heatmap(df.corr())
plt.show()


