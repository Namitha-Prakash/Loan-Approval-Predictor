import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os


def train():
    loan_dataset = pd.read_csv("loan.csv")

    loan_dataset = loan_dataset.dropna()

    loan_dataset = loan_dataset.replace(to_replace="3+", value="4")

    loan_dataset.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)
    loan_dataset.replace(
        {"Gender": {"Female": 0, "Male": 1}, "Married": {"Yes": 1, "No": 0}, "Self_Employed": {"Yes": 1, "No": 0},
         "Property_Area": {"Rural": 0, "Urban": 2, "Semiurban": 1}, "Education": {"Graduate": 1, "Not Graduate": 0}},
        inplace=True)
    X = loan_dataset.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
    Y = loan_dataset["Loan_Status"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on training data: ', training_data_accuracy)
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on test data: ', test_data_accuracy)

    # Save the trained model using pickle
    with open("loan_classifier_model.pkl", 'wb') as model_file:
        pickle.dump(classifier, model_file)


def predict_loan_approval(input_data):
    # Load the trained model
    with open("loan_classifier_model.pkl", 'rb') as model_file:
        classifier = pickle.load(model_file)

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = classifier.predict(input_data_as_numpy_array)

    if prediction == 0:
        return 'We regret to inform you your loan has not been approved.'
    else:
        return 'Congratulations, your loan has been approved!'


if __name__ == "__main__":
    # Check if the saved model exists
    if os.path.exists("loan_classifier_model.pkl"):
        with open("loan_classifier_model.pkl", 'rb') as model_file:
            classifier = pickle.load(model_file)
    else:
        # If the model doesn't exist, train it
        train()
        with open("loan_classifier_model.pkl", 'rb') as model_file:
            classifier = pickle
            load(model_file)

    name = input("Enter your name: ")
    gender = input("Enter Your Gender [Male:1, Female:0]: ")
    m = input("Enter your marital status [Yes:1, No:0]: ")
    d = input("Enter no. of people dependent on you [if 3 or more:4]: ")
    e = input("Enter your education [Graduated:1, not Graduated:0]: ")
    s = input("Enter your employment status [Yes:1, No:0]: ")
    a = float(input("Enter your income: "))

    coapplicant = input("Will there be a co-applicant? (Yes/No): ")
    if coapplicant.lower() == "yes":
        ca = float(input("Enter your coapplicant income: "))
    else:
        ca = 0  # Set coapplicant income to 0 if there is no co-applicant

    la = float(input("Enter Loan amount: "))
    lt = float(input("Enter loan term in terms of days: "))
    ch = input("Enter Your credit history [repaid all your previous loans on time: 1.0, if not: 0]: ")
    pa = input("Enter Property area [rural:0, Semiurban:1, Urban:2]")

    input_data = (gender, m, d, e, s, a, ca, la, lt, ch, pa)
    result = predict_loan_approval(input_data)
    print(result)
