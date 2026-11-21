
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df.drop('Play', axis=1)
y = df['Play']

model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

print("Decision Tree using ID3 Algorithm")
print("---------------------------------")
print(export_text(model, feature_names=list(X.columns)))

sample = pd.DataFrame(
    [[encoders['Outlook'].transform(['Sunny'])[0],
      encoders['Temperature'].transform(['Cool'])[0],
      encoders['Humidity'].transform(['High'])[0],
      encoders['Wind'].transform(['Strong'])[0]]],
    columns=X.columns
)

prediction = model.predict(sample)[0]

print("New Sample: Sunny, Cool, High, Strong")
print("Predicted Result:", encoders['Play'].inverse_transform([prediction])[0])
