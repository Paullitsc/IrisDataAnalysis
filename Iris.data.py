#Performed Scatterplot, Box plot, Sample Regression, and Descriptive statistics
#for Fisher's Iris data set made by Ronald Fisher in 1936
#Used Dictionarys:Pandas, matplot, and seaborn in Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_label']
iris_data = pd.read_csv(iris_url, names=iris_columns)

# Create a pairplot to visualize scatter plots for each pair of features
sns.pairplot(data=iris_data, hue='class_label')
plt.show()

#####################################################################################
# Custom Boxplots for Iris Dataset

# Extract unique classes from the 'class_label' column
unique_classes = iris_data['class_label'].unique()

# Generate a custom palette dynamically based on unique classes
class_palette = {label: sns.color_palette()[i] for i, label in enumerate(unique_classes)}

plt.figure(figsize=(12, 6))

# Boxplot for sepal length
plt.subplot(2, 2, 1)
sns.boxplot(x='class_label', y='sepal_length', data=iris_data, palette=class_palette)
plt.title('Sepal Length')

# Boxplot for sepal width
plt.subplot(2, 2, 2)
sns.boxplot(x='class_label', y='sepal_width', data=iris_data, palette=class_palette)
plt.title('Sepal Width')

# Boxplot for petal length
plt.subplot(2, 2, 3)
sns.boxplot(x='class_label', y='petal_length', data=iris_data, palette=class_palette)
plt.title('Petal Length')

# Boxplot for petal width
plt.subplot(2, 2, 4)
sns.boxplot(x='class_label', y='petal_width', data=iris_data, palette=class_palette)
plt.title('Petal Width')

plt.tight_layout()
plt.show()

########################################################################################
# Linear Regression for Petal Length and Width

# Linear regression for petal length based on petal width for each category
sns.lmplot(x='petal_width', y='petal_length', data=iris_data, hue='class_label', aspect=1.5, height=6)

# Linear regression for all data points
sns.regplot(x='petal_width', y='petal_length', data=iris_data, scatter=False, color='black', label='Overall Regression')

plt.legend()
plt.title('Regression Lines for Each Category and Overall')
plt.xlabel('Petal Width')
plt.ylabel('Petal Length')
plt.show()

########################################################################################
# Descriptive Statistics for Iris Dataset

# Load the Iris dataset for the last time
iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_label']
iris_data = pd.read_csv(iris_url, names=iris_columns)

# Calculate descriptive statistics for the dataset
descriptive_stats = iris_data.describe()

# Display the descriptive statistics
print(descriptive_stats)


