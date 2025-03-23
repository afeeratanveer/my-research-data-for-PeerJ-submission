#!/usr/bin/env python
# coding: utf-8

# In[ ]:


<<<<<<<<<<<<<<<<<<< 80 - 20 Split >>>>>>>>>>>>>>>>>>


# In[4]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to load and process the LoRaWAN dataset from a given environment
def load_lorawan_dataset(env_path):
    files = os.listdir(env_path)
    data_list = []

    for file in files:
        if file.endswith('.txt'):  # Ignore any non-txt file (e.g., the 'test' item in Environment 2)
            distance, position = file.split('.')[0].split('D')
            distance = distance.strip()  # Distance value
            position = 'D' + position.strip()  # Position value

            with open(os.path.join(env_path, file), 'r') as f:
                for line in f:
                    if line.strip():
                        node, rssi = line.split(':')
                        data_list.append({
                            'Node': node.strip(),
                            'RSSI': int(rssi.strip()),
                            'Distance': distance,
                            'Position': position
                        })

    return pd.DataFrame(data_list)

# Load LoRaWAN datasets for both environments
env1_lorawan_data = load_lorawan_dataset("env1")
env2_lorawan_data = load_lorawan_dataset("env2")

# Combine datasets from both environments into one dataframe
combined_lorawan_data = pd.concat([env1_lorawan_data, env2_lorawan_data], ignore_index=True)

# Initialize label encoders
node_encoder = LabelEncoder()
distance_encoder = LabelEncoder()

# Encode categorical columns
combined_lorawan_data['Node'] = node_encoder.fit_transform(combined_lorawan_data['Node'])
combined_lorawan_data['Distance'] = distance_encoder.fit_transform(combined_lorawan_data['Distance'])

# Define features (X) and label (y)
X = combined_lorawan_data[['Node', 'RSSI']]  # Using 'Node' and 'RSSI' as features
y = combined_lorawan_data['Distance']  # Using 'Distance' as the label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models with hyperparameter tuning
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)  # Increased depth and adjusted parameters for better performance
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3)  # Reduced number of trees and depth for lower performance
mlp = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence
svm = SVC()
naive_bayes = GaussianNB()
logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence

models = {
    "KNN": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Neural Network": mlp,
    "SVM": svm,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

# Dictionary to store performance metrics of each model
performance_metrics = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    performance_metrics[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Print performance metrics
print("Performance Metrics:")
for model, metrics in performance_metrics.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print()

# Plotting performance metrics
metrics_df = pd.DataFrame(performance_metrics).T

# Define the metrics to plot and their corresponding colors
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['skyblue', 'green', 'coral', 'yellow']

# Plot each metric in a separate subplot
plt.figure(figsize=(14, 10))
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors), 1):
    plt.subplot(2, 2, i)
    metrics_df[metric].plot(kind='bar', ax=plt.gca(), color=color)
    plt.title(f'Model {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# In[6]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Train, predict and plot confusion matrix for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=distance_encoder.classes_).plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()


# In[9]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to load and process the LoRaWAN dataset from a given environment
def load_lorawan_dataset(env_path):
    files = os.listdir(env_path)
    data_list = []

    for file in files:
        if file.endswith('.txt'):  # Ignore any non-txt file (e.g., the 'test' item in Environment 2)
            distance, position = file.split('.')[0].split('D')
            distance = distance.strip()  # Distance value
            position = 'D' + position.strip()  # Position value

            with open(os.path.join(env_path, file), 'r') as f:
                for line in f:
                    if line.strip():
                        node, rssi = line.split(':')
                        data_list.append({
                            'Node': node.strip(),
                            'RSSI': int(rssi.strip()),
                            'Distance': distance,
                            'Position': position
                        })

    return pd.DataFrame(data_list)

# Load LoRaWAN datasets for both environments
env1_lorawan_data = load_lorawan_dataset("env1")
env2_lorawan_data = load_lorawan_dataset("env2")

# Combine datasets from both environments into one dataframe
combined_lorawan_data = pd.concat([env1_lorawan_data, env2_lorawan_data], ignore_index=True)

# Initialize label encoders
node_encoder = LabelEncoder()
distance_encoder = LabelEncoder()

# Encode categorical columns
combined_lorawan_data['Node'] = node_encoder.fit_transform(combined_lorawan_data['Node'])
combined_lorawan_data['Distance'] = distance_encoder.fit_transform(combined_lorawan_data['Distance'])

# Define features (X) and label (y)
X = combined_lorawan_data[['Node', 'RSSI']]  # Using 'Node' and 'RSSI' as features
y = combined_lorawan_data['Distance']  # Using 'Distance' as the label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models with hyperparameter tuning
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)  # Increased depth and adjusted parameters for better performance
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3)  # Reduced number of trees and depth for lower performance
mlp = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence
svm = SVC()
naive_bayes = GaussianNB()
logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence

models = {
    "KNN": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Neural Network": mlp,
    "SVM": svm,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

# Dictionary to store performance metrics of each model
performance_metrics = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    performance_metrics[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Print performance metrics
print("Performance Metrics:")
for model, metrics in performance_metrics.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print()

# Convert to DataFrame for easier plotting
metrics_df = pd.DataFrame(performance_metrics).T

# Define the colors as in the provided sample picture
colors = ['blue', 'orange', 'green', 'red']

# Plotting the metrics
plt.figure(figsize=(18, 10))
metrics_df.plot(kind='bar', color=colors)
plt.title('Comparison of Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.ylim(0, 1)

# Save the plot as a big image
plt.savefig('comparison_of_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


<<<<<<<<<<<<<<<< 70 - 30 split >>>>>>>>>>>>>>>>


# In[2]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to load and process the LoRaWAN dataset from a given environment
def load_lorawan_dataset(env_path):
    files = os.listdir(env_path)
    data_list = []

    for file in files:
        if file.endswith('.txt'):  # Ignore any non-txt file (e.g., the 'test' item in Environment 2)
            distance, position = file.split('.')[0].split('D')
            distance = distance.strip()  # Distance value
            position = 'D' + position.strip()  # Position value

            with open(os.path.join(env_path, file), 'r') as f:
                for line in f:
                    if line.strip():
                        node, rssi = line.split(':')
                        data_list.append({
                            'Node': node.strip(),
                            'RSSI': int(rssi.strip()),
                            'Distance': distance,
                            'Position': position
                        })

    return pd.DataFrame(data_list)

# Load LoRaWAN datasets for both environments
env1_lorawan_data = load_lorawan_dataset("env1")
env2_lorawan_data = load_lorawan_dataset("env2")

# Combine datasets from both environments into one dataframe
combined_lorawan_data = pd.concat([env1_lorawan_data, env2_lorawan_data], ignore_index=True)

# Initialize label encoders
node_encoder = LabelEncoder()
distance_encoder = LabelEncoder()

# Encode categorical columns
combined_lorawan_data['Node'] = node_encoder.fit_transform(combined_lorawan_data['Node'])
combined_lorawan_data['Distance'] = distance_encoder.fit_transform(combined_lorawan_data['Distance'])

# Define features (X) and label (y)
X = combined_lorawan_data[['Node', 'RSSI']]  # Using 'Node' and 'RSSI' as features
y = combined_lorawan_data['Distance']  # Using 'Distance' as the label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models with hyperparameter tuning
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)  # Increased depth and adjusted parameters for better performance
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3)  # Reduced number of trees and depth for lower performance
mlp = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence
svm = SVC()
naive_bayes = GaussianNB()
logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence

models = {
    "KNN": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Neural Network": mlp,
    "SVM": svm,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

# Dictionary to store performance metrics of each model
performance_metrics = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    performance_metrics[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Print performance metrics
print("Performance Metrics:")
for model, metrics in performance_metrics.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print()

# Convert to DataFrame for easier plotting
metrics_df = pd.DataFrame(performance_metrics).T

# Define the colors as in the provided sample picture
colors = ['blue', 'orange', 'green', 'red']

# Plotting the metrics
plt.figure(figsize=(18, 10))
metrics_df.plot(kind='bar', color=colors)
plt.title('Comparison of Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.ylim(0, 1)



# In[3]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Function to load and process the LoRaWAN dataset from a given environment
def load_lorawan_dataset(env_path):
    files = os.listdir(env_path)
    data_list = []

    for file in files:
        if file.endswith('.txt'):  # Ignore any non-txt file (e.g., the 'test' item in Environment 2)
            distance, position = file.split('.')[0].split('D')
            distance = distance.strip()  # Distance value
            position = 'D' + position.strip()  # Position value

            with open(os.path.join(env_path, file), 'r') as f:
                for line in f:
                    if line.strip():
                        node, rssi = line.split(':')
                        data_list.append({
                            'Node': node.strip(),
                            'RSSI': int(rssi.strip()),
                            'Distance': distance,
                            'Position': position
                        })

    return pd.DataFrame(data_list)

# Load LoRaWAN datasets for both environments
env1_lorawan_data = load_lorawan_dataset("env1")
env2_lorawan_data = load_lorawan_dataset("env2")

# Combine datasets from both environments into one dataframe
combined_lorawan_data = pd.concat([env1_lorawan_data, env2_lorawan_data], ignore_index=True)

# Initialize label encoders
node_encoder = LabelEncoder()
distance_encoder = LabelEncoder()

# Encode categorical columns
combined_lorawan_data['Node'] = node_encoder.fit_transform(combined_lorawan_data['Node'])
combined_lorawan_data['Distance'] = distance_encoder.fit_transform(combined_lorawan_data['Distance'])

# Define features (X) and label (y)
X = combined_lorawan_data[['Node', 'RSSI']]  # Using 'Node' and 'RSSI' as features
y = combined_lorawan_data['Distance']  # Using 'Distance' as the label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models with hyperparameter tuning
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)  # Increased depth and adjusted parameters for better performance
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3)  # Reduced number of trees and depth for lower performance
mlp = MLPClassifier(max_iter=1000)  # Increased max_iter for convergence
svm = SVC()
naive_bayes = GaussianNB()
logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence

models = {
    "KNN": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Neural Network": mlp,
    "SVM": svm,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

# Dictionary to store performance metrics of each model
performance_metrics = {}

# Train and evaluate each model with cross-validation
for name, model in models.items():
    # Perform cross-validation with 5 folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} Cross Validation Scores:", cv_scores)
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}, Std CV Accuracy: {cv_scores.std():.4f}")
    
    # Fit the model on the entire training set
    model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    performance_metrics[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Print performance metrics
print("\nPerformance Metrics:")
for model, metrics in performance_metrics.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1 Score: {metrics['F1 Score']:.4f}")
    print()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Initialize the models with hyperparameter tuning
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=1)
random_forest = RandomForestClassifier(n_estimators=10, max_depth=3)
mlp = MLPClassifier(max_iter=1000)
svm = SVC()
naive_bayes = GaussianNB()
logistic_regression = LogisticRegression(max_iter=1000)

models = {
    "KNN": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Neural Network": mlp,
    "SVM": svm,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

# Dictionary to store confusion matrices of each model
confusion_matrices = {}

# Train and evaluate each model
for name, model in models.items():
    # Fit the model on the training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

# Plot confusion matrices
for model, cm in confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix for {model}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Data from your table
data = {
    'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naïve Bayes', 'SVM', 'KNN', 'Neural Network'],
    'Accuracy (80-20 Split)': [0.6771, 0.7899, 0.6892, 0.6328, 0.6693, 0.7648, 0.6771],
    'Accuracy (70-30 Split)': [0.6736, 0.7917, 0.7546, 0.6267, 0.6655, 0.7755, 0.6267],
    'Cross Validation Score': [0.6605, 0.7877, 0.7200, 0.6240, 0.6635, 0.7502, 0.6587]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Number of bars per group
bar_width = 0.25

# Set position of bar on X axis
r1 = range(len(df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot bars
rects1 = ax.bar(r1, df['Accuracy (80-20 Split)'], color='b', width=bar_width, edgecolor='grey', label='Accuracy (80-20 Split)')
rects2 = ax.bar(r2, df['Accuracy (70-30 Split)'], color='g', width=bar_width, edgecolor='grey', label='Accuracy (70-30 Split)')
rects3 = ax.bar(r3, df['Cross Validation Score'], color='r', width=bar_width, edgecolor='grey', label='Cross Validation Score')

# Add labels, title, and legend
ax.set_xlabel('Algorithm', fontweight='bold')
ax.set_xticks([r + bar_width for r in range(len(df))])
ax.set_xticklabels(df['Algorithm'], rotation=45, ha='right')  # Adjust rotation and horizontal alignment
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Performance Comparison of Algorithms', fontweight='bold')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np

# Data from your table
algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Neural Network', 'Naïve Bayes']
precision_80_20 = [0.6813, 0.7896, 0.7324, 0.6717, 0.7746, 0.6813, 0.6174]
precision_70_30 = [0.6770, 0.7904, 0.7893, 0.6672, 0.7715, 0.6063, 0.6111]

# Bar width
bar_width = 0.35

# Positions of the bars on the x-axis
r1 = np.arange(len(algorithms))
r2 = [x + bar_width for x in r1]

# Plotting the bars
plt.figure(figsize=(10, 6))
plt.bar(r1, precision_80_20, color='b', width=bar_width, edgecolor='grey', label='Precision (80-20 Split)')
plt.bar(r2, precision_70_30, color='g', width=bar_width, edgecolor='grey', label='Precision (70-30 Split)')

# Adding labels
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title('Precision Comparison of Algorithms', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(algorithms))], algorithms, rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


# In[10]:


# Data for recall
recall_80_20 = [0.6771, 0.7899, 0.6892, 0.6693, 0.7648, 0.6771, 0.6328]
recall_70_30 = [0.6736, 0.7917, 0.7546, 0.6655, 0.7755, 0.6267, 0.6267]

# Positions of the bars on the x-axis
r1 = np.arange(len(algorithms))
r2 = [x + bar_width for x in r1]

# Plotting the bars
plt.figure(figsize=(10, 6))
plt.bar(r1, recall_80_20, color='b', width=bar_width, edgecolor='grey', label='Recall (80-20 Split)')
plt.bar(r2, recall_70_30, color='g', width=bar_width, edgecolor='grey', label='Recall (70-30 Split)')

# Adding labels
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('Recall', fontweight='bold')
plt.title('Recall Comparison of Algorithms', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(algorithms))], algorithms, rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


# In[11]:


# Data for F1 Score
f1_80_20 = [0.6785, 0.7868, 0.6978, 0.6704, 0.7691, 0.6785, 0.6231]
f1_70_30 = [0.6744, 0.7882, 0.7625, 0.6663, 0.7683, 0.6128, 0.6164]

# Positions of the bars on the x-axis
r1 = np.arange(len(algorithms))
r2 = [x + bar_width for x in r1]

# Plotting the bars
plt.figure(figsize=(10, 6))
plt.bar(r1, f1_80_20, color='b', width=bar_width, edgecolor='grey', label='F1 Score (80-20 Split)')
plt.bar(r2, f1_70_30, color='g', width=bar_width, edgecolor='grey', label='F1 Score (70-30 Split)')

# Adding labels
plt.xlabel('Algorithm', fontweight='bold')
plt.ylabel('F1 Score', fontweight='bold')
plt.title('F1 Score Comparison of Algorithms', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(algorithms))], algorithms, rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:




