# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:21:29 2023

@author: DANIEL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA   
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.offline as pyo
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore")  

#MEDIDAS DE DISIMILARIDAD
#MEDIDAS DE DISIMILARIDAD
medidas = pd.read_excel("tennis_players.xlsx")
medidas = medidas.dropna()
medidas= medidas.sort_values(by="Desempeño")#ordené los sujetos por desempeño para poder analizar bien la matriz de disimilaridad
labels = medidas['Desempeño'] 
features = medidas.iloc[:, 4:63] #si quisiera extraerse directamente columnas

# obtengo los 2 dataframe
df1 = features.iloc[:,:32]
df = features.iloc[:, 50:62]
df = features.iloc[:, -1:]
antropo = pd.concat([df1, df], axis=1)
fisicas = features.iloc[:, 33:63]

##features
# Hacer escalamiento a los datos
scalar = preprocessing.StandardScaler()#normaliza  media 0 varianza 1. otros escaladores MinMaxScaler y MaxAbsScaler, 
scaled_data=pd.DataFrame(scalar.fit_transform(features))
#PCA
pca = PCA(n_components = 3)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
fig=px.scatter_3d(df_pca,x='PC1',y='PC2',z='PC3')
fig.show()
pyo.plot(fig, filename='3d_featuresPCA_plot.html') # all

# Matriz de disimilardiad
'''
pairwise=squareform(pdist(scaled_data,'euclidean'))
hm=sns.heatmap(pairwise)
'''


# Define the distance metrics
distance_metrics = ['cityblock', 'jaccard', 'hamming', 'euclidean']

# Create a figure with subplots for each dissimilarity matrix
fig, axes = plt.subplots(nrows=len(distance_metrics), figsize=(10, 8 * len(distance_metrics)))

# Iterate over the distance metrics
for i, metric in enumerate(distance_metrics):
    # Compute the pairwise distances using the specified metric
    pairwise = squareform(pdist(scaled_data, metric))
    
    # Plot the dissimilarity matrix
    ax = axes[i]
    sns.heatmap(pairwise, ax=ax, cmap='viridis')
    ax.set_title(f'Dissimilarity Matrix ({metric.capitalize()})')

plt.tight_layout()
plt.show()


#KPCA 

Kernel_pca = KernelPCA(kernel= "rbf",n_components=3)# extracts 2 features, specify the kernel as rbf
Z = Kernel_pca.fit_transform(scaled_data)
pairwise=squareform(pdist(Z))
hm=sns.heatmap(pairwise)


#features = medidas.iloc[:, 4:64] #si quisiera extraerse directamente columnas
features=features.dropna()
labels = features['Desempeño'] 
#labels2=labels.
sns.scatterplot(x=Z[:, 1], y=Z[:, 2],hue=labels)
Z = Kernel_pca .fit_transform(scaled_data)

# Create a DataFrame with the transformed data
df_kpca = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3'])

# Plot the 3D scatter plot with color-coded labels
fig = px.scatter_3d(df_kpca, x='PC1', y='PC2', z='PC3', color=labels)
fig.show()
pyo.plot(fig, filename='3d_KPCA_plot.html')

# validation


X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.4, random_state=0)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

# Create and fit the KNN classifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)

# Predict labels for the training and testing sets
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)

# Print accuracy scores
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

# Print classification report
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Testing Classification Report:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Perform cross-validation
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(neigh, features, labels, scoring=scoring, cv=5)

# Print cross-validation results
print("Cross-Validation Results:")
for metric in scoring:
    print(metric, ":", cv_results['test_' + metric])

   

# Create and fit the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Predict labels for the training and testing sets
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

# Print accuracy scores
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

# Print classification report
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Testing Classification Report:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Perform cross-validation
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(svm, features, labels, scoring=scoring, cv=5)

# Print cross-validation results
print("Cross-Validation Results:")
for metric in scoring:
    print(metric, ":", cv_results['test_' + metric])
    
print("Clasificación con PCA, arriba")
warnings.filterwarnings("ignore")  # Ignore warning messages

n_components = 3  # Number of components for Kernel PCA
kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # List of kernels to iterate over

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)

for kernel in kernels:
    # Create an instance of KernelPCA with the current kernel
    kpca = KernelPCA(kernel=kernel, n_components=n_components)

    # Fit and transform the data using KPCA
    Z = kpca.fit_transform(X_train)

    # Create a DataFrame with the transformed data
    df_kpca = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3'])

    # Plot the 3D scatter plot with color-coded labels
    fig = px.scatter_3d(df_kpca, x='PC1', y='PC2', z='PC3', color=y_train)
    fig.update_layout(title=f'Kernel PCA - {kernel} Kernel')
    fig.show()

    # Plot the 2D scatter plot with color-coded labels
    fig2 = px.scatter(df_kpca, x='PC1', y='PC2', color=y_train)
    fig2.update_layout(title=f'Kernel PCA - {kernel} Kernel (2D)')
    fig2.show()

    # Save the plots as HTML files
    pyo.plot(fig, filename=f'3d_KPCA_{kernel}_plot.html')
    pyo.plot(fig2, filename=f'2d_KPCA_{kernel}_plot.html')

    # Fit the KPCA on the training data
    Z_train = kpca.transform(X_train)

    # Fit the SVM classifier on the transformed training data
    svm = SVC()
    svm.fit(Z_train, y_train)

    # Transform the test data using KPCA
    Z_test = kpca.transform(X_test)

    # Predict labels for the training and testing sets
    y_train_pred = svm.predict(Z_train)
    y_test_pred = svm.predict(Z_test)

    # Print accuracy scores
    print("Training Accuracy (SVM):", accuracy_score(y_train, y_train_pred))
    print("Testing Accuracy (SVM):", accuracy_score(y_test, y_test_pred))

    # Print classification report
    print("Training Classification Report (SVM):")
    print(classification_report(y_train, y_train_pred))
    print("Testing Classification Report (SVM):")
    print(classification_report(y_test, y_test_pred))

    # Print confusion matrix
    print("Training Confusion Matrix (SVM):")
    print(confusion_matrix(y_train, y_train_pred))
    print("Testing Confusion Matrix (SVM):")
    print(confusion_matrix(y_test, y_test_pred))

    # Perform cross-validation with SVM
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results_svm = cross_validate(svm, Z_train, y_train, scoring=scoring, cv=5)

    # Print cross-validation results for SVM
    print("Cross-Validation Results (SVM):")
    for metric in scoring:
        print(metric, ":", cv_results_svm['test_' + metric])

    # Fit the KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(Z_train, y_train)

    # Predict labels for the training and testing sets
    y_train_pred_knn = knn.predict(Z_train)
    y_test_pred_knn = knn.predict(Z_test)

    # Print accuracy scores for KNN
    print("Training Accuracy (KNN):", accuracy_score(y_train, y_train_pred_knn))
    print("Testing Accuracy (KNN):", accuracy_score(y_test, y_test_pred_knn))

    # Print classification report for KNN
    print("Training Classification Report (KNN):")
    print(classification_report(y_train, y_train_pred_knn))
    print("Testing Classification Report (KNN):")
    print(classification_report(y_test, y_test_pred_knn))

    # Print confusion matrix for KNN
    print("Training Confusion Matrix (KNN):")
    print(confusion_matrix(y_train, y_train_pred_knn))
    print("Testing Confusion Matrix (KNN):")
    print(confusion_matrix(y_test, y_test_pred_knn))

    # Perform cross-validation with KNN
    cv_results_knn = cross_validate(knn, Z_train, y_train, scoring=scoring, cv=5)

    # Print cross-validation results for KNN
    print("Cross-Validation Results (KNN):")
    for metric in scoring:
        print(metric, ":", cv_results_knn['test_' + metric])

print("SVM:")
print("Mean Accuracy (SVM):", np.mean(cv_results_svm['test_accuracy']))
print("Mean Precision (SVM):", np.mean(cv_results_svm['test_precision_macro']))
print("Mean Recall (SVM):", np.mean(cv_results_svm['test_recall_macro']))
print("Mean F1-score (SVM):", np.mean(cv_results_svm['test_f1_macro']))
#print("Best Kernel (SVM):", best_kernel_svm)

print("\nKNN:")
print("Mean Accuracy (KNN):", np.mean(cv_results_knn['test_accuracy']))
print("Mean Precision (KNN):", np.mean(cv_results_knn['test_precision_macro']))
print("Mean Recall (KNN):", np.mean(cv_results_knn['test_recall_macro']))
print("Mean F1-score (KNN):", np.mean(cv_results_knn['test_f1_macro']))
#print("Best Kernel (KNN):", best_kernel_knn)

# Determine the best model and kernel based on the chosen evaluation metric
best_model = None
best_kernel_svm = None
best_kernel_knn = None
best_metric_value = -1

if np.mean(cv_results_svm['test_accuracy']) > best_metric_value:
    best_model = "SVM"
    best_metric_value = np.mean(cv_results_svm['test_accuracy'])
    best_kernel_svm = kernel

if np.mean(cv_results_knn['test_accuracy']) > best_metric_value:
    best_model = "KNN"
    best_metric_value = np.mean(cv_results_knn['test_accuracy'])

if np.mean(cv_results_svm['test_precision_macro']) > best_metric_value:
    best_model = "SVM"
    best_metric_value = np.mean(cv_results_svm['test_precision_macro'])
    best_kernel_svm = kernel

if np.mean(cv_results_knn['test_precision_macro']) > best_metric_value:
    best_model = "KNN"
    best_metric_value = np.mean(cv_results_knn['test_precision_macro'])

if np.mean(cv_results_svm['test_recall_macro']) > best_metric_value:
    best_model = "SVM"
    best_metric_value = np.mean(cv_results_svm['test_recall_macro'])
    best_kernel_svm = kernel

if np.mean(cv_results_knn['test_recall_macro']) > best_metric_value:
    best_model = "KNN"
    best_metric_value = np.mean(cv_results_knn['test_recall_macro'])

if np.mean(cv_results_svm['test_f1_macro']) > best_metric_value:
    best_model = "SVM"
    best_metric_value = np.mean(cv_results_svm['test_f1_macro'])
    best_kernel_svm = kernel

if np.mean(cv_results_knn['test_f1_macro']) > best_metric_value:
    best_model = "KNN"
    best_metric_value = np.mean(cv_results_knn['test_f1_macro'])

# Print the best model and kernel based on the chosen evaluation metric
print("\nBest Model:", best_model, "(Based on the chosen evaluation metric)")
if best_model == "SVM":
    print("Best Kernel:", best_kernel_svm)
elif best_model == "KNN":
    print("Best Kernel:", best_kernel_knn)