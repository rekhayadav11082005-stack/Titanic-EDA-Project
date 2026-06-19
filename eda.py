import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Age Distribution
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Survival Count
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.show()

# Passenger Class Count
df["Pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()