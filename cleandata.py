import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

df = pd.read_csv("datset.csv")

print(df.isnull().sum())

df_label = df.copy()

le = LabelEncoder()

df_label['gender_Encoded'] = le.fit_transform(df['gender'])

df_label['platform_Encoded'] = le.fit_transform(df['platform'])

df_label['interests_Encoded'] = le.fit_transform(df['interests'])

df_label['location_Encoded'] = le.fit_transform(df['location'])

df_label['demographics_Encoded'] = le.fit_transform(df['demographics'])

df_label['profession_Encoded'] = le.fit_transform(df['profession'])

df_label['indebt_Encoded'] = le.fit_transform(df['indebt'])

df_label['isHomeOwner_Encoded'] = le.fit_transform(df['isHomeOwner'])

df_label['Owns_Car_Encoded'] = le.fit_transform(df['Owns_Car'])

print(df_label)

df_label.to_csv("cleandata.csv",index=False)

scall = [
    'age',
    'time_spent',
    'income'
]

scaller = StandardScaler()

scal = scaller.fit_transform(df[scall])

scal_data = pd.DataFrame(scal,columns=scall)

print(scal_data)

df_final = pd.concat([scal_data,df_label],axis=1)

df_final.to_csv("finaldataset.csv",index=False)



