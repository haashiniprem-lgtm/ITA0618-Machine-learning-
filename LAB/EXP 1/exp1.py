
# FIND-S Algorithm

training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Initialize the most specific hypothesis
hypothesis = ['0'] * (len(training_data[0]) - 1)

print("FIND-S Algorithm")
print("----------------")

for example in training_data:
    if example[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'

        print("Updated Hypothesis:", hypothesis)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)
