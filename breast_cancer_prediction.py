import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_decision_boundary(model, X, y, title, is_proba=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    if hasattr(model, "predict_proba"):  
        Z = model.predict_proba(xy)[:, 1].reshape(XX.shape)  
        plt.contour(XX, YY, Z, levels=[0.5], linewidths=2, colors="black")
    else:
        Z = model.decision_function(xy).reshape(XX.shape)  
        plt.contour(XX, YY, Z, levels=[0], linewidths=2, colors="black")

    plt.title(title)
    plt.show()

# Support Vector Machine
linear_svm = SVC(kernel='linear', C=1, random_state=0)
linear_svm.fit(X_train, y_train)
train_predictions_svm = linear_svm.predict(X_train)
test_predictions_svm = linear_svm.predict(X_test)
cm_svm = confusion_matrix(y_test, test_predictions_svm)
train_accuracy_svm = accuracy_score(y_train, train_predictions_svm)
test_accuracy_svm = accuracy_score(y_test, test_predictions_svm)
print("SVM Confusion Matrix:\n", cm_svm)
print("SVM TRAIN Accuracy:", train_accuracy_svm)
print("SVM TEST Accuracy:", test_accuracy_svm)
plot_decision_boundary(linear_svm, X, y, "SVM Decision Boundary")

# Logistic Regression
logistic_reg = LogisticRegression(random_state=0)
logistic_reg.fit(X_train, y_train)
train_predictions_logreg = logistic_reg.predict(X_train)
test_predictions_logreg = logistic_reg.predict(X_test)
cm_logreg = confusion_matrix(y_test, test_predictions_logreg)
train_accuracy_logreg = accuracy_score(y_train, train_predictions_logreg)
test_accuracy_logreg = accuracy_score(y_test, test_predictions_logreg)
print("Logistic Regression Confusion Matrix:\n", cm_logreg)
print("Logistic Regression TRAIN Accuracy:", train_accuracy_logreg)
print("Logistic Regression TEST Accuracy:", test_accuracy_logreg)
plot_decision_boundary(logistic_reg, X, y, "Logistic Regression Decision Boundary")

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
train_predictions_rf = random_forest.predict(X_train)
test_predictions_rf = random_forest.predict(X_test)
cm_rf = confusion_matrix(y_test, test_predictions_rf)
train_accuracy_rf = accuracy_score(y_train, train_predictions_rf)
test_accuracy_rf = accuracy_score(y_test, test_predictions_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)
print("Random Forest TRAIN Accuracy:", train_accuracy_rf)
print("Random Forest TEST Accuracy:", test_accuracy_rf)
plot_decision_boundary(random_forest, X, y, "Random Forest Decision Boundary")
