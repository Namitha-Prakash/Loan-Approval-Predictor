import tkinter as tk
from tkinter import ttk
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


def submit_input():
    name = name_entry.get()
    gender = gender_var.get()
    marital_status = marital_status_var.get()
    dependents = dependents_var.get()
    education = education_var.get()
    employment_status = employment_status_var.get()
    income = float(income_entry.get())

    coapplicant = coapplicant_var.get()
    if coapplicant == "Yes":
        coapplicant_income = float(coapplicant_income_entry.get())
    else:
        coapplicant_income = 0

    loan_amount = float(loan_amount_entry.get())
    loan_term = float(loan_term_entry.get())
    credit_history = float(credit_history_var.get())
    property_area = property_area_var.get()

    input_data = (gender, marital_status, dependents, education, employment_status, income,
                  coapplicant_income, loan_amount, loan_term, credit_history, property_area)

    result = predict_loan_approval(input_data)
    result_label.config(text=result)


# Initialize the main application window
root = tk.Tk()
root.title("Loan Approval Prediction")

# Create and configure a frame
frame = ttk.Frame(root, padding=10)
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create and add labels and entry fields for each feature
name_label = ttk.Label(frame, text="Enter your name:")
name_label.grid(column=0, row=0, sticky=tk.W)
name_entry = ttk.Entry(frame)
name_entry.grid(column=1, row=0, columnspan=2)

# Create and add other input fields for features (e.g., gender, marital status, etc.)
gender_label = ttk.Label(frame, text="Gender [Male:1, Female:0]:")
gender_label.grid(column=0, row=1, sticky=tk.W)
gender_var = tk.StringVar(value="1")
gender_entry = ttk.Combobox(frame, textvariable=gender_var, values=["0", "1"])
gender_entry.grid(column=1, row=1)

marital_status_label = ttk.Label(frame, text="Marital Status [Yes:1, No:0]:")
marital_status_label.grid(column=0, row=2, sticky=tk.W)
marital_status_var = tk.StringVar(value="1")
marital_status_entry = ttk.Combobox(frame, textvariable=marital_status_var, values=["0", "1"])
marital_status_entry.grid(column=1, row=2)

dependents_label = ttk.Label(frame, text="Dependents [if 3 or more:4]:")
dependents_label.grid(column=0, row=3, sticky=tk.W)
dependents_var = tk.StringVar(value="0")
dependents_entry = ttk.Combobox(frame, textvariable=dependents_var, values=["0", "1", "2", "4"])
dependents_entry.grid(column=1, row=3)

education_label = ttk.Label(frame, text="Education [Graduated:1, not Graduated:0]:")
education_label.grid(column=0, row=4, sticky=tk.W)
education_var = tk.StringVar(value="1")
education_entry = ttk.Combobox(frame, textvariable=education_var, values=["0", "1"])
education_entry.grid(column=1, row=4)

employment_status_label = ttk.Label(frame, text="Employment Status [Yes:1, No:0]:")
employment_status_label.grid(column=0, row=5, sticky=tk.W)
employment_status_var = tk.StringVar(value="1")
employment_status_entry = ttk.Combobox(frame, textvariable=employment_status_var, values=["0", "1"])
employment_status_entry.grid(column=1, row=5)

income_label = ttk.Label(frame, text="Income:")
income_label.grid(column=0, row=6, sticky=tk.W)
income_entry = ttk.Entry(frame)
income_entry.grid(column=1, row=6, columnspan=2)

# Create and configure a radio button for co-applicant
coapplicant_label = ttk.Label(frame, text="Will there be a co-applicant?")
coapplicant_label.grid(column=0, row=7, sticky=tk.W)
coapplicant_var = tk.StringVar(value="No")
coapplicant_yes = ttk.Radiobutton(frame, text="Yes", variable=coapplicant_var, value="Yes")
coapplicant_yes.grid(column=1, row=7)
coapplicant_no = ttk.Radiobutton(frame, text="No", variable=coapplicant_var, value="No")
coapplicant_no.grid(column=2, row=7)

# Add entry field for co-applicant income
coapplicant_income_label = ttk.Label(frame, text="Coapplicant Income:")
coapplicant_income_label.grid(column=0, row=8, sticky=tk.W)
coapplicant_income_entry = ttk.Entry(frame)
coapplicant_income_entry.grid(column=1, row=8, columnspan=2)

# Create and configure entry fields for loan amount
loan_amount_label = ttk.Label(frame, text="Loan Amount:")
loan_amount_label.grid(column=0, row=9, sticky=tk.W)
loan_amount_entry = ttk.Entry(frame)
loan_amount_entry.grid(column=1, row=9, columnspan=2)

# Create and configure entry fields for loan term
loan_term_label = ttk.Label(frame, text="Loan Term (in days):")
loan_term_label.grid(column=0, row=10, sticky=tk.W)
loan_term_entry = ttk.Entry(frame)
loan_term_entry.grid(column=1, row=10, columnspan=2)

# Create and configure entry fields for credit history
credit_history_label = ttk.Label(frame, text="Credit History [repaid all previous loans on time: 1.0, if not: 0]:")
credit_history_label.grid(column=0, row=11, sticky=tk.W)
credit_history_var = tk.StringVar(value="1.0")
credit_history_entry = ttk.Combobox(frame, textvariable=credit_history_var, values=["0", "1.0"])
credit_history_entry.grid(column=1, row=11)

# Create and configure entry fields for property area
property_area_label = ttk.Label(frame, text="Property Area [rural:0, Semiurban:1, Urban:2]:")
property_area_label.grid(column=0, row=12, sticky=tk.W)
property_area_var = tk.StringVar(value="1")
property_area_entry = ttk.Combobox(frame, textvariable=property_area_var, values=["0", "1", "2"])
property_area_entry.grid(column=1, row=12)

# Create and configure a submit button
submit_button = ttk.Button(frame, text="Submit", command=submit_input)
submit_button.grid(column=0, row=13, columnspan=3)

# Create a label to display the prediction result
result_label = ttk.Label(frame, text="")
result_label.grid(column=0, row=14, columnspan=3)

# Start the GUI application
root.mainloop()
