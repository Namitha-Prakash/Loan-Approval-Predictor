# Loan Approval Predictor

This is a **Loan Approval Predictor** built using **Support Vector Machines (SVM)** to predict whether a loan will be approved or not based on several input factors. The project uses user input and machine learning to assess the likelihood of loan approval.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

## Features

- Predicts loan approval based on user-provided factors such as income, employment status, education, and more.
- Uses **Support Vector Machine (SVM)** for classification.
- User-friendly command-line interface for entering data.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/loan-approval-predictor.git
    cd loan-approval-predictor
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python loan_approval_predictor.py
    ```

## Usage

After installing the dependencies and running the script, you will be prompted to enter the following details:

```text
Enter your name:
Enter Your Gender [Male:1, Female:0]:
Enter your marital status [Yes:1, No:0]:
Enter number of people dependent on you [if 3 or more:4]:
Enter your education status [Graduated:1, not Graduated:0]:
Enter your employment status [Yes:1, No:0]:
Enter your income:
