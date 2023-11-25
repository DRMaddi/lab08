import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load data from Excel file
file_path = r"C:\Users\anish\Downloads\Lab3\embeddingsdata.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")  # Assuming your data is in the first sheet

X = df.loc[:, 'embed_1':'embed_76'].values
y = df['Label'].values  

# Split Your Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the Right Naïve Bayes Model
model = GaussianNB()

# Train the Naïve Bayes Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Additional Evaluation Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
