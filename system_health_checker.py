import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create simple sample data
data = {
    "cpu_usage": [20, 35, 40, 75, 80, 90, 50, 60, 85, 30],
    "memory_usage": [30, 40, 45, 70, 75, 85, 50, 55, 80, 35],
    "temperature": [40, 45, 50, 75, 80, 90, 55, 60, 85, 42],
    "crash_count": [0, 0, 1, 3, 4, 5, 1, 2, 4, 0],
    "risk": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
}

# Step 2: Convert data into table format
df = pd.DataFrame(data)

# Step 3: Select input and output
X = df[["cpu_usage", "memory_usage", "temperature", "crash_count"]]
y = df["risk"]

# Step 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create model
model = DecisionTreeClassifier()

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Test model
y_pred = model.predict(X_test)

# Step 8: Show accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Check a new system
new_data = pd.DataFrame([{
    "cpu_usage": 82,
    "memory_usage": 78,
    "temperature": 84,
    "crash_count": 4
}])

result = model.predict(new_data)

# Step 10: Print final result
if result[0] == 1:
    print("System is at risk")
else:
    print("System is healthy")
