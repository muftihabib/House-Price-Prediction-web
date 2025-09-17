import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import json

# --- 1. Load the Data ---
print("Loading data...")
df = pd.read_csv("cleaned_bengaluru_house_data.csv")


# --- 2. Feature Engineering & Outlier Removal ---
print("Performing feature engineering and outlier removal...")

# Create the 'price_per_sqft' column
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Group rare locations into 'other'
df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
locations_less_than_10 = location_stats[location_stats <= 10]
df.location = df.location.apply(lambda x: 'other' if x in locations_less_than_10 else x)

# Remove outliers where sqft per bhk is less than 300
df = df[~(df.total_sqft / df.bhk < 300)]


# --- 3. Prepare Data for Model Training ---
# Use one-hot encoding for the 'location' column
dummies = pd.get_dummies(df.location)
df = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')

# Drop the original 'location' column and the 'price_per_sqft' column
df = df.drop('location', axis='columns')
df = df.drop('price_per_sqft', axis='columns')

# Define features (X) and target (y)
X = df.drop(['price'], axis='columns')
y = df.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# --- 4. Train the Model ---
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
print(f"Model accuracy (R^2 score): {model.score(X_test, y_test):.2f}")


# --- 5. Save the Model and Columns ---
print("Saving model and artifacts...")
# Save the trained model to a pickle file
with open('bangalore_house_price_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Save the column names to a json file
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

print("Script finished successfully!")