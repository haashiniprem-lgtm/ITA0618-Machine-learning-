
import csv

# Create training dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Save dataset into CSV file
with open('training_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sky', 'AirTemp', 'Humidity', 'Wind',
                     'Water', 'Forecast', 'EnjoySport'])
    writer.writerows(data)

# Read CSV file
with open('training_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    training_data = list(reader)

num_attributes = len(training_data[0]) - 1

S = ['0'] * num_attributes
G = [['?'] * num_attributes]

for example in training_data:
    attributes = example[:-1]
    label = example[-1]

    if label == 'Yes':
        G = [g for g in G if all(
            g[i] == '?' or g[i] == attributes[i]
            for i in range(num_attributes)
        )]

        for i in range(num_attributes):
            if S[i] == '0':
                S[i] = attributes[i]
            elif S[i] != attributes[i]:
                S[i] = '?'

    else:
        new_G = []
        for g in G:
            for i in range(num_attributes):
                if g[i] == '?':
                    if S[i] != '?' and S[i] != '0':
                        new_hypothesis = g.copy()
                        new_hypothesis[i] = S[i]
                        new_G.append(new_hypothesis)

        G = new_G

print("Candidate-Elimination Algorithm")
print("--------------------------------")
print("Specific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")
for hypothesis in G:
    print(hypothesis)
