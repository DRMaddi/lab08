import pandas as pd

# Create a DataFrame from the data
data = {
    'Class': ['Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer'],
}

df = pd.DataFrame(data)

# Calculate the prior probability for each class
class_counts = df['Class'].value_counts()
prior_probabilities = class_counts / df.shape[0]

print(prior_probabilities)
