import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sklearn.utils as utils
import sklearn.preprocessing as preprocessing
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('Student_performance_prediction/Data.csv')

# Graph plotting and analysis
choice = 0
while(choice != 10):
    print("1. Class Count Graph\t2. Semester-wise Graph\n3. Gender-wise Graph\t4. Nationality-wise Graph\n5. Grade-wise Graph\t6. Section-wise Graph\n7. Topic-wise Graph\t8. Stage-wise Graph\n9. Absent Days-wise Graph\t10. No Graph\n")
    choice = int(input("Enter Choice: "))
    if (choice == 10):
        print("Exiting..\n")
        time.sleep(1)
        break
    print("Loading Graph....\n")
    time.sleep(1)
    if choice == 1:
        print("\tClass Count Graph")
        axes = sns.countplot(x='Class', data=data, order=['L', 'M', 'H'])
    elif choice == 2:
        print("\tSemester-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 3:
        print("\tGender-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 4:
        print("\tNationality-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 5:
        print("\tGrade-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 6:
        print("\tSection-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 7:
        print("\tTopic-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 8:
        print("\tStage-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    elif choice == 9:
        print("\tAbsent Days-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=axesarr)
    plt.show()

# Data preprocessing
data = data.drop(columns=['gender', 'StageID', 'GradeID', 'NationalITy', 'PlaceofBirth', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentschoolSatisfaction', 'ParentAnsweringSurvey', 'AnnouncementsView'])
utils.shuffle(data)

# Encoding categorical variables
for column in data.columns:
    if data[column].dtype == 'object':
        label_encoder = preprocessing.LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])

# Splitting data into training and testing sets
index = int(len(data) * 0.70)
features = data.iloc[:, 0:4].values
labels = data.iloc[:, 4].values
features_train = features[:index]
features_test = features[index + 1:]
labels_train = labels[:index]
labels_test = labels[index + 1:]

# Model training and evaluation
model_decision_tree = tree.DecisionTreeClassifier()
model_decision_tree.fit(features_train, labels_train)
labels_pred_decision_tree = model_decision_tree.predict(features_test)
accuracy_decision_tree = metrics.accuracy_score(labels_test, labels_pred_decision_tree)
print("\nAccuracy measures using Decision Tree:")
print(metrics.classification_report(labels_test, labels_pred_decision_tree),"\n")
print("Accuracy using Decision Tree:", round(accuracy_decision_tree, 3))

time.sleep(1)
model_random_forest = ensemble.RandomForestClassifier()
model_random_forest.fit(features_train, labels_train)
labels_pred_random_forest = model_random_forest.predict(features_test)
accuracy_random_forest = metrics.accuracy_score(labels_test, labels_pred_random_forest)
print("\nAccuracy measures for Random Forest Classifier: \n")
print(metrics.classification_report(labels_test, labels_pred_random_forest))
print("Accuracy using Random Forest:", round(accuracy_random_forest, 3))

time.sleep(1)
model_perceptron = linear_model.Perceptron()
model_perceptron.fit(features_train, labels_train)
labels_pred_perceptron = model_perceptron.predict(features_test)
accuracy_perceptron = metrics.accuracy_score(labels_test, labels_pred_perceptron)
print("\nAccuracy measures using Linear Model Perceptron:")
print(metrics.classification_report(labels_test, labels_pred_perceptron),"\n") 
print("Accuracy using Linear Model Perceptron:", round(accuracy_perceptron, 3), "\n")

time.sleep(1)
model_logistic_regression = linear_model.LogisticRegression()
model_logistic_regression.fit(features_train, labels_train)
labels_pred_logistic_regression = model_logistic_regression.predict(features_test)
accuracy_logistic_regression = metrics.accuracy_score(labels_test, labels_pred_logistic_regression)
print("\nAccuracy measures using Linear Model Logistic Regression:")
print(metrics.classification_report(labels_test, labels_pred_logistic_regression),"\n")
print("Accuracy using Linear Model Logistic Regression:", round(accuracy_logistic_regression, 3), "\n")

time.sleep(1)
model_mlp = neural_network.MLPClassifier(activation="logistic")
model_mlp.fit(features_train, labels_train)
labels_pred_mlp = model_mlp.predict(features_test)
accuracy_mlp = metrics.accuracy_score(labels_test, labels_pred_mlp)
print("\nAccuracy measures using MLP Classifier:")
print(metrics.classification_report(labels_test, labels_pred_mlp),"\n")
print("Accuracy using Neural Network MLP Classifier:", round(accuracy_mlp, 3), "\n")

# Testing specific input
choice = input("Do you want to test specific input (y or n): ")
if choice.lower() == "y":
    gender = input("Enter Gender (M or F): ")
    gender = 1 if gender.upper() == "M" else 0
    nationality = input("Enter Nationality: ")
    place_of_birth = input("Place of Birth: ")
    grade_id = input("Grade ID as (G-<grade>): ")
    grade_id = int(grade_id.split("-")[1])
    section = input("Enter Section: ")
    topic = input("Enter Topic: ")
    semester = input("Enter Semester (F or S): ")
    semester = 0 if semester.upper() == "F" else 1
    relation = input("Enter Relation (Father or Mum): ")
    relation = 0 if relation == "Father" else 1
    raised_hands = int(input("Enter raised hands: "))
    visited_resources = int(input("Enter Visited Resources: "))
    announcements_viewed = int(input("Enter announcements viewed: "))
    discussions = int(input("Enter no. of Discussions: "))
    parent_answered_survey = input("Enter Parent Answered Survey (Y or N): ")
    parent_answered_survey = 1 if parent_answered_survey.upper() == "Y" else 0
    parent_school_satisfaction = input("Enter Parent School Satisfaction (Good or Bad): ")
    parent_school_satisfaction = 1 if parent_school_satisfaction == "Good" else 0
    absences = input("Enter No. of Abscenes (Under-7 or Above-7): ")
    absences = 1 if absences == "Under-7" else 0
    input_data = np.array([raised_hands, visited_resources, discussions, absences])
    predictions_decision_tree = model_decision_tree.predict(input_data.reshape(1, -1))
    predictions_random_forest = model_random_forest.predict(input_data.reshape(1, -1))
    predictions_perceptron = model_perceptron.predict(input_data.reshape(1, -1))
    predictions_logistic_regression = model_logistic_regression.predict(input_data.reshape(1, -1))
    predictions_mlp = model_mlp.predict(input_data.reshape(1, -1))
    print("\nUsing Decision Tree Classifier:", predictions_decision_tree[0])
    time.sleep(1)
    print("Using Random Forest Classifier:", predictions_random_forest[0])
    time.sleep(1)
    print("Using Linear Model Perceptron:", predictions_perceptron[0])
    time.sleep(1)
    print("Using Linear Model Logisitic Regression:", predictions_logistic_regression[0])
    time.sleep(1)
    print("Using Neural Network MLP Classifier:", predictions_mlp[0])
    print("\nExiting...")
    time.sleep(1)
else:
    print("Exiting..")
    time.sleep(1)
