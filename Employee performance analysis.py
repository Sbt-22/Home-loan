import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


file_path = "/home/seabata/Documents/Data_Analysis_Project/loanSanctionTrain.csv" # Replace this with the actual path to your CSV file
data = pd.read_csv(file_path) # Load the CSV file
print(data.head())
print(data.info())
print(data.describe())

columns_to_fill_categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status' ] #Fill missing values
for column in columns_to_fill_categorical:
    data[column].fillna(data[column].mode()[0], inplace=True) # Fill missing values with the mode of each column using a loop

columns_to_fill_numerical = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'] #Fill in missing values alternatively to using a loop
modes = {column: data[column].mode()[0] for column in columns_to_fill_numerical}

data['Log_ApplicantIncome'] = np.log1p(data['ApplicantIncome']) #normalizing outliers which lead to high variance by applying log
data['Log_LoanAmount'] = np.log1p(data['LoanAmount'])
data['Log_CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
""""
sns.boxplot(x='Loan_Status', y='Log_ApplicantIncome', data=data)
plt.show()
sns.histplot(data['LoanAmount'], kde=True)
plt.show()
"""


label_enc = LabelEncoder()
data['Gender'] = label_enc.fit_transform(data['Gender'])
data['Married'] = label_enc.fit_transform(data['Married'])
data['Education'] = label_enc.fit_transform(data['Education'])
data['Self_Employed'] = label_enc.fit_transform(data['Self_Employed'])
data['Property_Area'] = label_enc.fit_transform(data['Property_Area'])
data['Loan_Status'] = label_enc.fit_transform(data['Loan_Status'])

X_train = data.drop(['Loan_Status', 'Loan_ID'], axis=1) #axis=1 means this is column op. If we equate it to 0, it is a row op.
y_train = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
'''
# Assuming 'data' is your DataFrame
data = pd.get_dummies(data, drop_first=True)

correlation = data['ApplicantIncome'].corr(data['Loan_Status'])
print("Correlation between ApplicantIncome and Loan_Status:", correlation)

# Compute and plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
'''

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

