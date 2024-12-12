# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Loading the .XPT File and converting it csv
file_path_xpt = r'C:\Users\matth\Downloads\DPQ_L.XPT'
data = pd.read_sas(file_path_xpt)
csv_file_path = r'C:\Users\matth\Downloads\DPQ_L.csv'
data.to_csv(csv_file_path, index=False)
print(f"File saved to {csv_file_path}")

# Reload CSV
data = pd.read_csv(csv_file_path)

# Selecting Relevant Columns
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

# Data Cleaning
data = data.dropna()
# Change values to 0
data = data.applymap(lambda x: 0 if isinstance(x, (float, int)) and abs(x) < 1e-10 else x)
# Cap all values at 3
dpq_columns = [col for col in data.columns if col.startswith('DPQ') and col != 'DPQ100_cat']
data[dpq_columns] = data[dpq_columns].applymap(lambda x: min(x, 3) if isinstance(x, (float, int)) else x)

# Bin DPQ100 into categorical classes
data['DPQ100_cat'] = pd.cut(data['DPQ100'], bins=[-1, 0, 1, 2, 3], labels=['None', 'Low', 'Moderate', 'High'])

cleaned_csv_path = r'C:\Users\matth\Downloads\Cleaned_DPQ_L.csv'
data.to_csv(cleaned_csv_path, index=False)

# Exploratory Data Analysis
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

# Target and Features for Classification
X = data.drop(['SEQN', 'DPQ100', 'DPQ100_cat'], axis=1)  # Features (excluding ID and original DPQ100)
y = data['DPQ100_cat']  # Target variable as categorical

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values with the mean
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Ensure y_train and y_test do not contain NaN values
y_train = y_train.dropna()
y_test = y_test.dropna()

# Align X_train and y_train
X_train, y_train = X_train.align(y_train, join='inner', axis=0)
# Align X_test and y_test
X_test, y_test = X_test.align(y_test, join='inner', axis=0)

# Build and Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top Features in Predicting Difficulty Due to Mental Health Symptoms (DPQ100)")
plt.show()