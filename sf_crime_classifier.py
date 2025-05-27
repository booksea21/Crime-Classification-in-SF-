
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import geopandas as gpd
import folium

df = pd.read_csv("san_train.csv", parse_dates=["Dates"])

df = df.drop(columns=["Address", "Descript"], errors="ignore")

df["location_count"] = df.groupby(["X", "Y"])["Dates"].transform("count")
df["Month"] = df["Dates"].dt.month_name()
df["DayOfWeek"] = pd.Categorical(df["DayOfWeek"])
df["Category"] = pd.Categorical(df["Category"])

df.dropna(inplace=True)

X = df.drop(columns=["Category", "Dates"])
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5, ccp_alpha=0.005, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Decision Tree for Crime Classification")
plt.show()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Month", hue="Category", multiple="fill", shrink=0.8)
plt.title("Crime Distribution by Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="DayOfWeek", hue="Category", multiple="fill", shrink=0.8)
plt.title("Crime Distribution by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
