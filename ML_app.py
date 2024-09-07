import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# app title
st.title("Iris Species Classification using Decision Tree")
st.write("""
This application allows you to classify iris species using a decision tree model.
""")


iris = datasets.load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

np.random.seed(42)
iris_df.loc[np.random.choice(iris_df.index, 10), 'sepal length (cm)'] = np.nan

imputer = SimpleImputer(strategy='mean')
iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']] = imputer.fit_transform(
    iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
)

scaler = MinMaxScaler()
iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']] = scaler.fit_transform(
    iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
)

X = iris_df.drop(columns=['species'])
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.write(f"### Model Evaluation:")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
st.pyplot(fig)

classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)
st.write(f"### Classification Report:")
st.text(classification_rep)

st.sidebar.header("Predict Iris Species")
sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
new_data_normalized = scaler.transform(new_data)

predicted_class = clf.predict(new_data_normalized)
predicted_species = iris.target_names[predicted_class][0]

st.sidebar.write(f"## Predicted Species: **{predicted_species}**")
