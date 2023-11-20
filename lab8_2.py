import pandas as pd

# Create a DataFrame from the data
data = {
    'Age': [25, 26, 35, 30, 22, 37, 56, 28, 27, 52, 31],
    'Income': [56000, 82000, 67000, 75000, 53000, 70000, 25000, 48000, 38000, 64000, 51000],
    'Student': ['Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
    'Credit rating': ['Excellent', 'Good', 'Fair', 'Excellent', 'Fair', 'Excellent', 'Poor', 'Good', 'Fair', 'Excellent', 'Fair'],
    'Class': ['Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Buys computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer', 'Does not buy computer'],
}

df = pd.DataFrame(data)

# Calculate the class conditional densities for each feature
class_conditional_densities = {}
for feature in df.columns[:-1]:
    for class_label in df['Class'].unique():
        class_conditional_densities[(feature, class_label)] = df[df['Class'] == class_label][feature].value_counts(normalize=True)

# Print the class conditional densities
for feature, class_label in class_conditional_densities:
    print(f"Class conditional density for {feature} = {class_label}:")
    print(class_conditional_densities[(feature, class_label)])
