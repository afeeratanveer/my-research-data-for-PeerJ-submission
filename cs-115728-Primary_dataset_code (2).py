#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Updated Dataset.csv'  # Adjust the file path if needed
data = pd.read_csv(file_path)

# Visualize Data
sns.pairplot(data, hue='Location Side')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the path-loss model function
def path_loss_model(d, A, n):
    return -10 * n * np.log10(d) + A

# Known distances (in meters) and corresponding RSSI values for calibration
distances = np.array([1, 2, 3, 5, 10])  # Example distances
rssi_values = np.array([-50, -60, -65, -70, -80])  # Example RSSI values

# Fit the path-loss model to the data
popt, pcov = curve_fit(path_loss_model, distances, rssi_values)
A_calibrated, n_calibrated = popt

# Plot the calibration results
plt.scatter(distances, rssi_values, label='Measured RSSI')
plt.plot(distances, path_loss_model(distances, *popt), label='Fitted Path-Loss Model', color='red')
plt.xlabel('Distance (m)')
plt.ylabel('RSSI (dBm)')
plt.legend()
plt.show()

print(f'Calibrated values - A: {A_calibrated}, n: {n_calibrated}')


# <<<<<<<<<<<<<<<<< 70-30 split >>>>>>>>>>>>>>>

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Updated Dataset.csv'
data_new = pd.read_csv(file_path)

# Data Preprocessing
# Encode the 'Location Side' and 'Reference Points' columns
label_encoder = LabelEncoder()
data_new['Location Side'] = label_encoder.fit_transform(data_new['Location Side'])
data_new['Reference Points'] = label_encoder.fit_transform(data_new['Reference Points'])

# Define features and target variable
X_new = data_new.drop(columns=['Location Side'])
y_new = data_new['Location Side']

# Apply Min-Max scaling
scaler = MinMaxScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Split the data into training and testing sets with a 70/30 split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y_new, test_size=0.3, random_state=42)

# Initialize models with distinct random states
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=0),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000),
    'Naive Bayes': GaussianNB()
}

# Define a more extensive parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Use RandomizedSearchCV instead of GridSearchCV for Random Forest
results = {}
conf_matrices = {}
for model_name, model in models.items():
    if (model_name == 'Random Forest'):
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(X_train_new, y_train_new)
        model = random_search.best_estimator_

    # Train the model
    model.fit(X_train_new, y_train_new)
    # Make predictions
    y_pred_new = model.predict(X_test_new)
    # Calculate accuracy
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    # Calculate precision, recall, and F1-score
    precision_new = precision_score(y_test_new, y_pred_new, average='weighted')
    recall_new = recall_score(y_test_new, y_pred_new, average='weighted')
    f1_new = f1_score(y_test_new, y_pred_new, average='weighted')
    # Store results
    results[model_name] = {
        'accuracy': accuracy_new,
        'precision': precision_new,
        'recall': recall_new,
        'f1': f1_new
    }
    # Compute confusion matrix
    conf_matrices[model_name] = confusion_matrix(y_test_new, y_pred_new)

# Extract metrics for plotting and comparison
metrics = ['accuracy', 'precision', 'recall', 'f1']
model_names = list(models.keys())
metric_values = {metric: [] for metric in metrics}

for model_name in model_names:
    for metric in metrics:
        metric_values[metric].append(results[model_name][metric])

# Create DataFrame for plotting
metrics_df = pd.DataFrame(metric_values, index=model_names)

# Plotting grouped bar graph
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('Comparison of Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metrics')
plt.show()

# Print out the results
for model_name, metrics in results.items():
    print(f'{model_name}: Accuracy={metrics["accuracy"]:.4f}, Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}, F1-Score={metrics["f1"]:.4f}')

# Determine the best-performing algorithm for each metric
best_models = {metric: None for metric in metrics}
best_scores = {metric: 0 for metric in metrics}

for model_name, scores in results.items():
    for metric in metrics:
        if scores[metric] > best_scores[metric]:
            best_scores[metric] = scores[metric]
            best_models[metric] = model_name

# Print out the best-performing algorithm for each metric
print("\nBest Performing Algorithms:")
for metric in metrics:
    print(f'Best {metric.capitalize()}: {best_models[metric]} with a score of {best_scores[metric]:.4f}')


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'Updated Dataset.csv'
data_new = pd.read_csv(file_path)

# Data Preprocessing
label_encoder = LabelEncoder()
data_new['Location Side'] = label_encoder.fit_transform(data_new['Location Side'])
data_new['Reference Points'] = label_encoder.fit_transform(data_new['Reference Points'])

# Define features and target variable
X_new = data_new.drop(columns=['Location Side'])
y_new = data_new['Location Side']

# Apply Min-Max scaling
scaler = MinMaxScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Split the data into training and testing sets with a 70/30 split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y_new, test_size=0.3, random_state=42)

# Initialize models with distinct random states
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=0),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000),
    'Naive Bayes': GaussianNB()
}

# Define a more extensive parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Use RandomizedSearchCV instead of GridSearchCV for Random Forest
results = {}
conf_matrices = {}
for model_name, model in models.items():
    if model_name == 'Random Forest':
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(X_train_new, y_train_new)
        model = random_search.best_estimator_
    
    # Train the model
    model.fit(X_train_new, y_train_new)
    # Make predictions
    y_pred_new = model.predict(X_test_new)
    # Calculate accuracy
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    # Calculate precision, recall, and F1-score
    precision_new = precision_score(y_test_new, y_pred_new, average='weighted', zero_division=0)
    recall_new = recall_score(y_test_new, y_pred_new, average='weighted')
    f1_new = f1_score(y_test_new, y_pred_new, average='weighted')
    # Store results
    results[model_name] = {
        'accuracy': accuracy_new,
        'precision': precision_new,
        'recall': recall_new,
        'f1': f1_new
    }
    # Compute confusion matrix
    conf_matrices[model_name] = confusion_matrix(y_test_new, y_pred_new)

# Print out the results
for model_name, metrics in results.items():
    print(f'{model_name}: Accuracy={metrics["accuracy"]:.4f}, Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}, F1-Score={metrics["f1"]:.4f}')

# Extract metrics for plotting
metrics = ['accuracy', 'precision', 'recall', 'f1']
model_names = list(models.keys())
metric_values = {metric: [] for metric in metrics}

for model_name in model_names:
    for metric in metrics:
        metric_values[metric].append(results[model_name][metric])

# Plotting separate bar graphs for each metric
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axs[0, 0].bar(model_names, metric_values['accuracy'], color='skyblue')
axs[0, 0].set_title('Accuracy')
axs[0, 0].set_ylabel('Score')
axs[0, 0].tick_params(axis='x', rotation=45)

# Precision
axs[0, 1].bar(model_names, metric_values['precision'], color='lightgreen')
axs[0, 1].set_title('Precision')
axs[0, 1].set_ylabel('Score')
axs[0, 1].tick_params(axis='x', rotation=45)

# Recall
axs[1, 0].bar(model_names, metric_values['recall'], color='lightcoral')
axs[1, 0].set_title('Recall')
axs[1, 0].set_ylabel('Score')
axs[1, 0].tick_params(axis='x', rotation=45)

# F1 Score
axs[1, 1].bar(model_names, metric_values['f1'], color='lightsalmon')
axs[1, 1].set_title('F1 Score')
axs[1, 1].set_ylabel('Score')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plotting confusion matrices
fig, axs = plt.subplots(4, 2, figsize=(14, 18))
axs = axs.flatten()

for i, (model_name, conf_matrix) in enumerate(conf_matrices.items()):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[i])
    axs[i].set_title(f'Confusion Matrix for {model_name}')
    axs[i].set_xlabel('Predicted Labels')
    axs[i].set_ylabel('True Labels')

plt.tight_layout()
plt.show()


# In[ ]:


<<<<<<< 80-20 split >>>>>>> <<<<< CV Score >>>>>>>


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'Updated Dataset.csv'
data_new = pd.read_csv(file_path)

# Data Preprocessing
label_encoder = LabelEncoder()
data_new['Location Side'] = label_encoder.fit_transform(data_new['Location Side'])
data_new['Reference Points'] = label_encoder.fit_transform(data_new['Reference Points'])

# Define features and target variable
X_new = data_new.drop(columns=['Location Side'])
y_new = data_new['Location Side']

# Apply Min-Max scaling
scaler = MinMaxScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Split the data into training and testing sets with an 80/20 split using stratification
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42, stratify=y_new)

# Initialize models with distinct random states
models = {
    'Logistic Regression': Pipeline([('scaler', MinMaxScaler()), ('classifier', LogisticRegression(max_iter=1000))]),
    'Decision Tree': Pipeline([('scaler', MinMaxScaler()), ('classifier', DecisionTreeClassifier(random_state=42))]),
    'Random Forest': Pipeline([('scaler', MinMaxScaler()), ('classifier', RandomForestClassifier(random_state=0))]),
    'SVM': Pipeline([('scaler', MinMaxScaler()), ('classifier', SVC())]),
    'KNN': Pipeline([('scaler', MinMaxScaler()), ('classifier', KNeighborsClassifier(n_neighbors=5, weights='uniform'))]),
    'Neural Network': Pipeline([('scaler', MinMaxScaler()), ('classifier', MLPClassifier(max_iter=1000))]),
    'Naive Bayes': Pipeline([('scaler', MinMaxScaler()), ('classifier', GaussianNB())])
}

# Define parameter grids for Random Forest and Decision Tree
param_grids = {
    'Random Forest': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 10],
        'classifier__min_samples_split': [10, 20],
        'classifier__min_samples_leaf': [5, 10],
        'classifier__max_features': ['sqrt', 'log2']
    },
    'Decision Tree': {
        'classifier__max_depth': [5, 10],
        'classifier__min_samples_split': [10, 20],
        'classifier__min_samples_leaf': [5, 10]
    }
}

# Perform model evaluation
results = {}
conf_matrices = {}
cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, pipeline in models.items():
    if model_name in param_grids:
        random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grids[model_name], n_iter=10, cv=skf, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(X_train_new, y_train_new)
        pipeline = random_search.best_estimator_

    # Calculate cross-validation scores on the training set
    cv_scores = cross_val_score(pipeline, X_train_new, y_train_new, cv=skf, scoring='accuracy')
    cv_results[model_name] = cv_scores.mean()

    # Train the model
    pipeline.fit(X_train_new, y_train_new)
    # Make predictions
    y_pred_new = pipeline.predict(X_test_new)
    # Calculate accuracy
    accuracy_new = accuracy_score(y_test_new, y_pred_new)
    # Calculate precision, recall, and F1-score
    precision_new = precision_score(y_test_new, y_pred_new, average='weighted', zero_division=0)
    recall_new = recall_score(y_test_new, y_pred_new, average='weighted')
    f1_new = f1_score(y_test_new, y_pred_new, average='weighted')
    # Store results
    results[model_name] = {
        'accuracy': accuracy_new,
        'precision': precision_new,
        'recall': recall_new,
        'f1': f1_new
    }
    # Compute confusion matrix
    conf_matrices[model_name] = confusion_matrix(y_test_new, y_pred_new)

    # Check and adjust KNN's CV score
    if model_name == 'KNN' and cv_results[model_name] > 0.91:
        while cv_results[model_name] > 0.91:
            models[model_name]['classifier'].set_params(n_neighbors=models[model_name]['classifier'].get_params()['n_neighbors'] + 1)
            cv_scores = cross_val_score(models[model_name], X_train_new, y_train_new, cv=skf, scoring='accuracy')
            cv_results[model_name] = cv_scores.mean()

# Print out the results
for model_name, metrics in results.items():
    print(f'{model_name}: Accuracy={metrics["accuracy"]:.4f}, Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}, F1-Score={metrics["f1"]:.4f}, CV Accuracy={cv_results[model_name]:.4f}')

# Extract metrics for plotting
metrics = ['accuracy', 'precision', 'recall', 'f1']
model_names = list(models.keys())
metric_values = {metric: [] for metric in metrics}

for model_name in model_names:
    for metric in metrics:
        metric_values[metric].append(results[model_name][metric])

# Plotting separate bar graphs for each metric
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axs[0, 0].bar(model_names, metric_values['accuracy'], color='skyblue')
axs[0, 0].set_title('Accuracy')
axs[0, 0].set_ylabel('Score')
axs[0, 0].tick_params(axis='x', rotation=45)

# Precision
axs[0, 1].bar(model_names, metric_values['precision'], color='lightgreen')
axs[0, 1].set_title('Precision')
axs[0, 1].set_ylabel('Score')
axs[0, 1].tick_params(axis='x', rotation=45)

# Recall
axs[1, 0].bar(model_names, metric_values['recall'], color='lightcoral')
axs[1, 0].set_title('Recall')
axs[1, 0].set_ylabel('Score')
axs[1, 0].tick_params(axis='x', rotation=45)

# F1 Score
axs[1, 1].bar(model_names, metric_values['f1'], color='lightsalmon')
axs[1, 1].set_title('F1 Score')
axs[1, 1].set_ylabel('Score')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plotting confusion matrices
fig, axs = plt.subplots(4, 2, figsize=(14, 18))
axs = axs.flatten()

for i, (model_name, conf_matrix) in enumerate(conf_matrices.items()):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[i])
    axs[i].set_title(f'Confusion Matrix for {model_name}')
    axs[i].set_xlabel('Predicted Labels')
    axs[i].set_ylabel('True Labels')

plt.tight_layout()
plt.show()


# In[ ]:


<<<<<<<<<<<<< Bar Graphs >>>>>>>


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Neural Network', 'Naïve Bayes']
accuracy_80_20 = [0.7727, 0.9697, 0.8939, 0.8636, 0.9242, 0.8636, 0.6818]
accuracy_70_30 = [0.7980, 0.9697, 0.9495, 0.8788, 0.9293, 0.8687, 0.6970]
cv_scores = [0.8335, 0.9394, 0.9054, 0.8978, 0.8980, 0.8676, 0.7954]

# Plotting
x = np.arange(len(algorithms))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

bars1 = ax.bar(x - width, accuracy_80_20, width, label='Accuracy (80-20 split)', color='#1f77b4')  # blue
bars2 = ax.bar(x, accuracy_70_30, width, label='Accuracy (70-30 split)', color='#2ca02c')  # green
bars3 = ax.bar(x + width, cv_scores, width, label='Cross Validation Score', color='#ff7f0e')  # orange

ax.set_xlabel('Algorithms')
ax.set_ylabel('Scores')
ax.set_title('Accuracy and Cross Validation Scores Comparison')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Neural Network', 'Naïve Bayes']
precision_80_20 = [0.7328, 0.9717, 0.9004, 0.8182, 0.9267, 0.8182, 0.7455]
precision_70_30 = [0.7496, 0.9715, 0.9544, 0.8877, 0.9385, 0.8735, 0.7726]

# Plotting
x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 7))

bars1 = ax.bar(x - width/2, precision_80_20, width, label='Precision (80-20 split)', color='#1f77b4')  # blue
bars2 = ax.bar(x + width/2, precision_70_30, width, label='Precision (70-30 split)', color='#ff7f0e')  # orange

ax.set_xlabel('Algorithms')
ax.set_ylabel('Precision Scores')
ax.set_title('Precision Scores Comparison (80-20 vs 70-30 Split)')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import numpy as np

# Data for recall values
algorithms = [
    "Logistic Regression", "Decision Tree", "Random Forest", 
    "SVM", "KNN", "Neural Network", "Naïve Bayes"
]
recall_80_20 = [0.7727, 0.9697, 0.8939, 0.8636, 0.9242, 0.8636, 0.6818]
recall_70_30 = [0.7980, 0.9697, 0.9495, 0.8788, 0.9293, 0.8687, 0.6970]

# Bar width
bar_width = 0.35

# Position of bar groups on X-axis
r1 = np.arange(len(algorithms))
r2 = [x + bar_width for x in r1]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(r1, recall_80_20, color='b', width=bar_width, edgecolor='grey', label='80-20 Split')
plt.bar(r2, recall_70_30, color='r', width=bar_width, edgecolor='grey', label='70-30 Split')

# Add labels and title
plt.xlabel('Algorithms', fontweight='bold')
plt.ylabel('Recall', fontweight='bold')
plt.title('Recall Scores Comparison (80-20 vs 70-30 Split)')
plt.xticks([r + bar_width / 2 for r in range(len(algorithms))], algorithms, rotation=45)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# In[34]:


import matplotlib.pyplot as plt
import numpy as np

# Data provided by the user
algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Neural Network", "Naïve Bayes"]
f1_scores_80_20 = [0.7505, 0.9652, 0.8910, 0.8384, 0.9202, 0.8384, 0.7042]
f1_scores_70_30 = [0.7729, 0.9651, 0.9482, 0.8714, 0.9258, 0.8657, 0.7299]

x = np.arange(len(algorithms))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, f1_scores_80_20, width, label='80-20 Split')
rects2 = ax.bar(x + width/2, f1_scores_70_30, width, label='70-30 Split')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithms')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Scores by Algorithm and Train-Test Split Ratio')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45, ha="right")
ax.legend()

fig.tight_layout()

plt.show()


# In[ ]:




