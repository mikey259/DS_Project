import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

#This function is to plot the confusion matrix.
def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

#Load the dataframe
from js import fetch
import io

URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = await fetch(URL1)
text1 = io.BytesIO((await resp1.arrayBuffer()).to_py())
data = pd.read_csv(text1)

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = await fetch(URL2)
text2 = io.BytesIO((await resp2.arrayBuffer()).to_py())
X = pd.read_csv(text2)

###task 1 Create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y,make sure the output is a Pandas series (only one bracket df['name of column']).
Y = pd.Series(data['Class'].to_numpy())

####task 2 Standardize the data in X then reassign it to the variable X using the transform provided below.
# students get this
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

####task 3 Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2.
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

####task 4Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
# Create the GridSearchCV object with the specified parameters
logreg_cv = GridSearchCV(lr, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to the data (replace X_train and y_train with your actual data)
logreg_cv.fit(X_train, y_train)

#We output the GridSearchCV object for logistic regression. We display the best parameters using the data attribute best_params_ and the accuracy on the validation data using the data attribute best_score_.
print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


#####task 5 Calculate the accuracy on the test data using the method score:
# Get the best model from GridSearchCV
best_model = logreg_cv.best_estimator_

# Calculate accuracy on the test data
test_accuracy = best_model.score(X_test, y_test)

print("Test Accuracy:", test_accuracy)

#Lets look at the confusion matrix:
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)



####task 6 Create a support vector machine object then create a GridSearchCV object svm_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
# Create GridSearchCV object
svm_cv = GridSearchCV(svm, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to the training data
svm_cv.fit(X_train, y_train)

# Print the best parameters and accuracy
print("Tuned hyperparameters (best parameters): ", svm_cv.best_params_)
print("Accuracy:", svm_cv.best_score_)


###task 7: Calculate the accuracy on the test data using the method score:
# Get the best model from GridSearchCV
best_svm_model = svm_cv.best_estimator_

# Calculate accuracy on the test data
test_accuracy = best_svm_model.score(X_test, y_test)

print("Test Accuracy:", test_accuracy)
#We can plot the confusion matrix
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


###task 8: Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
# Create GridSearchCV object
tree_cv = GridSearchCV(tree, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to the training data
tree_cv.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


#####task 9: Calculate the accuracy of tree_cv on the test data using the method score:
# Get the best model from GridSearchCV
best_tree_model = tree_cv.best_estimator_

# Calculate accuracy on the test data
test_accuracy = best_tree_model.score(X_test, y_test)

print("Test Accuracy:", test_accuracy)
#We can plot the confusion matrix
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


####task 10: Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
# Create GridSearchCV object
knn_cv = GridSearchCV(KNN, param_grid=parameters, cv=10)

# Fit the GridSearchCV object to the training data
knn_cv.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

####task 11: Calculate the accuracy of knn_cv on the test data using the method score:
# Get the best model from GridSearchCV
best_knn_model = knn_cv.best_estimator_

# Calculate accuracy on the test data
test_accuracy = best_knn_model.score(X_test, y_test)

print("Test Accuracy:", test_accuracy)
#We can plot the confusion matrix
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


####task 12: Find the method performs best:
# Calculate test accuracy for each model
#logreg_test_accuracy = logreg_cv.best_estimator_.score(X_test, y_test)
#svm_test_accuracy = svm_cv.best_estimator_.score(X_test, y_test)
#tree_test_accuracy = tree_cv.best_estimator_.score(X_test, y_test)
#knn_test_accuracy = knn_cv.best_estimator_.score(X_test, y_test)

# Print the accuracies
#print("Logistic Regression Test Accuracy:", logreg_test_accuracy)
#print("SVM Test Accuracy:", svm_test_accuracy)
#print("Decision Tree Test Accuracy:", tree_test_accuracy)
#print("KNN Test Accuracy:", knn_test_accuracy)

# Determine the best model
#best_model = max(logreg_test_accuracy, svm_test_accuracy, tree_test_accuracy, knn_test_accuracy)
#print("Best Model:", best_model)


#this method returns the model name by storing the names and scores in a dictionary, rather than just the score returned
model_accuracies = {
    "Logistic Regression": logreg_test_accuracy,
    "SVM": svm_test_accuracy,
    "Decision Tree": tree_test_accuracy,
    "KNN": knn_test_accuracy
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)
