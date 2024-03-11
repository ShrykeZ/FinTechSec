import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine

# Connect to the SQL database
database_url = 'testdb1.yo'
engine = create_engine(database_url)

# SQL query to fetch relevant data
query = 'SELECT amount, merchant, transaction_type, is_fraud FROM transactions;' # Example based on my test DB.
df = pd.read_sql(query, engine)

# Dataset columns 'c1', 'c2', ..., 'target'
features = df[['c1', 'c2']]
target = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, predictions))
