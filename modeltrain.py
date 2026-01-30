import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("finaldataset.csv")

x = df[['age_scal','gender_Encoded','time_spent_scal','platform_Encoded','demographics_Encoded','location_Encoded','profession_Encoded','income_scal']]
y = df[['isHomeOwner_Encoded']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=15)


model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train,y_train)

pre = model.predict(x_test)
score = accuracy_score(y_test,pre)
print("accuracy :",score*100)
