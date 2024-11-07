# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Load the .XPT File
file_path_xpt = r'C:\Users\matth\Downloads\DPQ_L.XPT'  # Path to your .XPT file
data = pd.read_sas(file_path_xpt)

# Step 2: Convert and Save to CSV
csv_file_path = r'C:\Users\matth\Downloads\DPQ_L.csv'  # Path to save the CSV file
data.to_csv(csv_file_path, index=False)
print(f"File saved to {csv_file_path}")

# Step 3: Reload CSV for Analysis (if needed)
data = pd.read_csv(csv_file_path)

# Step 4: Select Relevant Columns
selected_columns = [
    'SEQN',     # Respondent identifier
    'DPQ010',   # Little interest or pleasure in doing things
    'DPQ020',   # Feeling down, depressed, or hopeless
    'DPQ030',   # Trouble falling or staying asleep, or sleeping too much
    'DPQ040',   # Feeling tired or having little energy
    'DPQ050',   # Poor appetite or overeating
    'DPQ060',   # Feeling bad about yourself or feeling like a failure
    'DPQ070',   # Trouble concentrating on things
    'DPQ080',   # Moving or speaking slowly or being fidgety/restless
    'DPQ090',   # Thoughts of self-harm
    'DPQ100'    # Difficulty caused by these problems
]
data = data[selected_columns]

# Step 5: Data Cleaning: Drop Missing Values
data = data.dropna()

# Step 6: Convert DPQ100 to Categorical Levels
# Bin DPQ100 into categorical classes
data['DPQ100_cat'] = pd.cut(data['DPQ100'], bins=[-1, 0, 1, 2, 3], labels=['None', 'Low', 'Moderate', 'High'])

# Step 7: Exploratory Data Analysis (EDA)
# Plot distribution of "Feeling down, depressed, or hopeless" (DPQ020)
sns.histplot(data['DPQ020'], kde=True)
plt.title("Distribution of 'Feeling down, depressed, or hopeless'")
plt.show()

# Analyze "Little interest or pleasure in doing things" (DPQ010) by self-reported difficulty (DPQ100)
sns.boxplot(x='DPQ100_cat', y='DPQ010', data=data)
plt.title("Interest in Activities by Self-Reported Difficulty")
plt.xlabel("Self-Reported Difficulty (DPQ100_cat)")
plt.ylabel("Interest in Activities (DPQ010)")
plt.show()

# Step 8: Define Target and Features for Classification
X = data.drop(['SEQN', 'DPQ100', 'DPQ100_cat'], axis=1)  # Features (excluding ID and original DPQ100)
y = data['DPQ100_cat']  # Target variable as categorical

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Check for missing values
print("Missing values in X_train:\n", X_train.isnull().sum())
print("Missing values in X_test:\n", X_test.isnull().sum())

# Step 11: Handle missing values
# Option 1: Drop rows with missing values
# X_train = X_train.dropna()
# X_test = X_test.dropna()

# Option 2: Fill missing values with the mean (or median, mode, etc.)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Ensure y_train and y_test do not contain NaN values
y_train = y_train.dropna()
y_test = y_test.dropna()

# Step 12: Ensure consistent lengths
# Align X_train and y_train
X_train, y_train = X_train.align(y_train, join='inner', axis=0)
# Align X_test and y_test
X_test, y_test = X_test.align(y_test, join='inner', axis=0)

# Verify the lengths
print(f"Length of X_train: {len(X_train)}, Length of y_train: {len(y_train)}")
print(f"Length of X_test: {len(X_test)}, Length of y_test: {len(y_test)}")

# Step 13: Build and Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 14: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 15: Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top Features in Predicting Difficulty Due to Mental Health Symptoms (DPQ100)")
plt.show()
