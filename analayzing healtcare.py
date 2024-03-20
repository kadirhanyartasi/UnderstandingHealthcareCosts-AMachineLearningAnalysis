import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the insurance dataset into a DataFrame
df=pd.read_csv('insurance.csv')

# Display basic information about the dataset
df.info()

# Display descriptive statistics of the dataset
df.describe()

# Get the column names of the dataset
df.columns

# Select columns of interest for analysis
columns = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]

# Visualize categorical variables using pie charts
plt.figure(figsize = (22, 25))
for i, col in enumerate(columns):
    plt.subplot(4, 2, i+1)
    counts = df[col].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title(col)

plt.tight_layout()
plt.show()


# Count the occurrences of each BMI value
bmi_table = df['bmi'].value_counts()
print(bmi_table)

# Convert BMI values to categorical labels
df['bmi'] = df['bmi'].astype(float)

# Function to classify BMI into categories

def classify_bmi(bmi):
    if 18 <= bmi < 48:
        return 1
    elif 25.0 <= bmi < 29.9:
        return 2
    elif 30.0 <= bmi < 34.9:
        return 3
    else:
        return 4
# Apply the classification function to 'bmi' column

df['bmi'] = df['bmi'].apply(classify_bmi)

# Convert age values to categorical labels
df['age'] = df['age'].astype(float)

# Function to classify age into categories

def classify_age(age):
    if 18 <= age <= 27:
        return 1
    elif 28 <= age <= 39:
        return 2
    elif 40 <= age <= 51:
        return 3
    else:
        return 4
# Apply the classification function to 'age' column

df['age'] = df['age'].apply(classify_age)

# Count the occurrences of each number of children
children_table=df['children'].value_counts()
print(children_table)

# Convert number of children to categorical labels
df['children'] = df['children'].astype(float)

# Function to classify number of children into categories
def classify_children(children):
    if 0 <= children <= 1:
        return 1
    elif 2 <= children < 3:
        return 2
    else:
        return 3
    
# Apply the classification function to 'children' column
df['children'] = df['children'].apply(classify_children)

# Convert categorical variables to numerical labels
df['sex'] = df["sex"].map({"male": 1, "female": 0})
df['smoker'] = df["smoker"].map({"yes": 1, "no": 0})
df['region'] = df["region"].map({"southwest": 1, "southeast": 0,"northwest":2,"northeast":3})

# Display the first 10 rows of the modified DataFrame
df.head(10)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model using Random Forest Regressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(score)
from sklearn.linear_model import Ridge

# Assuming df, X, and y are defined similarly as in your code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge()  # Using Ridge regression instead of RandomForestRegressor
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(score)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming df, X, and y are defined similarly as in your code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()  # Using Linear Regression for regression
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(score)

from sklearn.ensemble import GradientBoostingRegressor

# Assuming df, X, and y are defined similarly as in your code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()  # Using GradientBoostingRegressor instead of RandomForestRegressor
model.fit(X_train, y_train)
score=model.score(X_test, y_test)
print(score)