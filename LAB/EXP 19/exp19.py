
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Training dataset
data = {
    'Income': ['High', 'Low', 'Medium', 'High', 'Low',
               'Medium', 'High', 'Low', 'Medium', 'High',
               'Low', 'Medium'],

    'CreditHistory': ['Good', 'Bad', 'Good', 'Good', 'Bad',
                      'Good', 'Good', 'Bad', 'Bad', 'Good',
                      'Bad', 'Good'],

    'Employment': ['Stable', 'Unstable', 'Stable', 'Stable',
                   'Unstable', 'Stable', 'Stable', 'Unstable',
                   'Unstable', 'Stable', 'Unstable', 'Stable'],

    'LoanStatus': ['Approved', 'Rejected', 'Approved', 'Approved',
                   'Rejected', 'Approved', 'Approved', 'Rejected',
                   'Rejected', 'Approved', 'Rejected', 'Approved']
}

df = pd.DataFrame(data)

# Encode categorical data
encoders = {}

for column in df.columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# Split features and target
X = df.drop('LoanStatus', axis=1)
y = df['LoanStatus']

# Train model
model = CategoricalNB()
model.fit(X, y)

# New customer details
sample = pd.DataFrame([[
    encoders['Income'].transform(['High'])[0],
    encoders['CreditHistory'].transform(['Good'])[0],
    encoders['Employment'].transform(['Stable'])[0]
]], columns=X.columns)

# Predict loan status
prediction = model.predict(sample)[0]

print("Bank Loan Prediction using Naive Bayes")
print("--------------------------------------")
print("Income: High")
print("Credit History: Good")
print("Employment: Stable")
print("Loan Status:",
      encoders['LoanStatus'].inverse_transform([prediction])[0])
