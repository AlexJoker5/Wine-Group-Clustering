import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Wine.csv")
df.head()
df.info()
df.nunique()
df.shape
df.describe()
X=df.drop(columns=['Customer_Segment','Proline', 'Magnesium'])

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=0)

kmeans.fit(X)

cluster_assignments = kmeans.labels_
final_centroids = kmeans.cluster_centers_

print("Final Cluster Assignments:", cluster_assignments)
print("Final Centroids:", final_centroids)

df['Cluster']=cluster_assignments
df.head()

df

X=X
y=cluster_assignments

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

df

new_data=[[13.10,1.70,2.50,12.5,2.80,3.20,0.25,2.00,6.15,0.98,3.50]]
pred = perceptron.predict(new_data)
print(pred)