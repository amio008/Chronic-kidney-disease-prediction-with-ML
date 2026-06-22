# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

#cols = ["id","age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane","classification"]

kidneycsv = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/422/Project/kidney_disease.csv')

kidneycsv.head()

print((kidneycsv.isnull().sum(axis=1))) #un

"""Finding possible outcomes for each attribute

---


"""

kidneycsv["age"].unique()

kidneycsv["bp"].unique()

kidneycsv["sg"].unique()

kidneycsv["al"].unique()

kidneycsv["su"].unique()

kidneycsv["rbc"].unique()

kidneycsv["pc"].unique()

kidneycsv["pcc"].unique()

kidneycsv["ba"].unique()

kidneycsv["bgr"].unique()

kidneycsv["bu"].unique()

kidneycsv["sc"].unique()

kidneycsv["sod"].unique()

kidneycsv["pot"].unique()

kidneycsv["hemo"].unique()

kidneycsv["pcv"].unique()

kidneycsv["wc"].unique()

kidneycsv["rc"].unique()

kidneycsv["htn"].unique()

kidneycsv["dm"].unique()

kidneycsv["cad"].unique()

kidneycsv["appet"].unique()

kidneycsv["pe"].unique()

kidneycsv["ane"].unique()

kidneycsv["classification"].unique()

"""#**Only nan as extra: rbc,pc,pcc,ba,htn,appet,pe,ane**"""

kidneycsv["rbc"] = kidneycsv["rbc"].apply(lambda x: 1 if x == "normal" else (0 if x == "abnormal" else np.nan))

kidneycsv["pc"] = kidneycsv["pc"].apply(lambda x: 1 if x == "normal" else (0 if x == "abnormal" else np.nan))

kidneycsv["pcc"] = kidneycsv["pcc"].apply(lambda x: 1 if x == "present" else (0 if x == "notpresent" else np.nan))

kidneycsv["ba"] = kidneycsv["ba"].apply(lambda x: 1 if x == "present" else (0 if x == "notpresent" else np.nan))

kidneycsv["htn"] = kidneycsv["htn"].apply(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))

kidneycsv["appet"] = kidneycsv["appet"].apply(lambda x: 1 if x == "good" else (0 if x == "poor" else np.nan))

kidneycsv["pe"] = kidneycsv["pe"].apply(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))

kidneycsv["ane"] = kidneycsv["ane"].apply(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))

"""#Washing corrupted values"""

kidneycsv["dm"] = kidneycsv["dm"].str.strip()

kidneycsv["cad"] = kidneycsv["cad"].str.strip()

kidneycsv["classification"] = kidneycsv["classification"].str.strip()

kidneycsv["rc"] = kidneycsv["rc"].apply(lambda x: np.nan if not str(x).replace('.', '', 1).isdigit() else float(x))

kidneycsv["wc"] = kidneycsv["wc"].apply(lambda x: np.nan if not str(x).replace('.', '', 1).isdigit() else float(x))

kidneycsv["pcv"] = kidneycsv["pcv"].apply(lambda x: np.nan if not str(x).replace('.', '', 1).isdigit() else float(x))

kidneycsv["classification"].unique()

kidneycsv["rc"].unique()

kidneycsv["wc"].unique()

kidneycsv["pcv"].unique()

kidneycsv["dm"].unique()

kidneycsv["cad"].unique()

"""#Converting numeric vals for the washed ones(Nan still left)"""

kidneycsv["dm"] = kidneycsv["dm"].apply(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))

kidneycsv["cad"] = kidneycsv["cad"].apply(lambda x: 1 if x == "yes" else (0 if x == "no" else np.nan))

kidneycsv["classification"] = kidneycsv["classification"].apply(lambda x: 1 if x == "ckd" else (0 if x == "notckd" else np.nan))

"""#Checking"""

kidneycsv.info() #un

kidneycsv.shape

kidneycsv.isnull().sum()

#kidneycsv['nan_count'] = kidneycsv.isnull().sum(axis=1)
print((kidneycsv.isnull().sum(axis=1)))

kidneycsv.head()

threshold=10
kidneycsv_rem_nan=kidneycsv[kidneycsv.isnull().sum(axis=1) <=threshold]

kidneycsv_rem_nan.isnull().sum()

kidneycsv_rem_nan.shape

kidneycsv_rem_nan.isnull().sum(axis=1) #un

kidneycsv_rem_nan[['age']]

kidneycsv_rem_nan["pcv"].unique()

kidneycsv_rem_nan.head()

kidneycsv_rem_nan.isnull().sum()

from sklearn.impute import SimpleImputer

float_cols = ['sg','sc','hemo','rc']
int_cols = ['age','bp','al','su','bgr','bu','sod','pot','pcv','wc']
impute = SimpleImputer(missing_values=np.nan, strategy='mean')

for col in int_cols:
    kidneycsv_rem_nan[[col]] = impute.fit_transform(kidneycsv_rem_nan[[col]])
    kidneycsv_rem_nan[col] = kidneycsv_rem_nan[col].astype(int)

for col in float_cols:
  kidneycsv_rem_nan[[col]] = impute.fit_transform(kidneycsv_rem_nan[[col]])


    #if kidneycsv_rem_nan[col].dtype == 'float64' and kidneycsv_rem_nan[col].dropna().apply(float.is_integer).all():

kidneycsv_rem_nan

kidneycsv_rem_nan

# Check for NaN values in specific columns
kidneycsv_rem_nan.isnull().sum()

big_nan_cols = ['rbc','pc']
small_nan_cols = ['pcc','ba','htn','dm','cad','appet','pe','ane']

impute_small = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

for col in small_nan_cols:
  impute_small.fit(kidneycsv_rem_nan[[col]])
  kidneycsv_rem_nan[col] = impute_small.transform(kidneycsv_rem_nan[[col]])

kidneycsv_rem_nan.isnull().sum()

kidneycsv_rem_nan.to_csv('new1.csv', index=False) #un

from google.colab import files

files.download('new1.csv')

i = 0
rows_checked = 0

while rows_checked < 243:
  if i in kidneycsv_rem_nan.index:
    if kidneycsv_rem_nan.loc[i, 'rbc'] not in [0, 1]:
      kidneycsv_rem_nan.loc[i, 'rbc'] = np.random.choice([0, 1])
    rows_checked += 1
  i += 1

while i < len(kidneycsv_rem_nan):
    if pd.isnull(kidneycsv_rem_nan.loc[i, 'rbc']):
        kidneycsv_rem_nan.loc[i, 'rbc'] = 1
    i += 1

i = 0
rows_checked = 0

while rows_checked < 243:
  if i in kidneycsv_rem_nan.index:
    if kidneycsv_rem_nan.loc[i, 'pc'] not in [0, 1]:
      kidneycsv_rem_nan.loc[i, 'pc'] = np.random.choice([0, 1])
    rows_checked += 1
  i += 1

while i < len(kidneycsv_rem_nan):
    if pd.isnull(kidneycsv_rem_nan.loc[i, 'pc']):
        kidneycsv_rem_nan.loc[i, 'pc'] = 1
    i += 1

kidneycsv_rem_nan.isnull().sum()

kidneycsv_rem_nan["rbc"].unique()

"""#Correlation heatmap"""

model_corr= kidneycsv_rem_nan.corr()
model_corr

import seaborn as sns

sns.heatmap(model_corr,cmap='YlGnBu')

corr_wrt_classification = model_corr["classification"]
drop_cols = corr_wrt_classification[(corr_wrt_classification >= -0.25)&(corr_wrt_classification<=0.25)].index
print(drop_cols)
kidneycsv_rem_nan = kidneycsv_rem_nan.drop(columns=drop_cols)
kidneycsv_rem_nan

"""#Data Shuffling"""

num_rows = len(kidneycsv_rem_nan)

new_order = []
for i in range((num_rows + 1) // 2):
    new_order.append(i)  # Add the next row from the top
    if i != num_rows - i - 1:  # Avoid duplicating the middle row in case of odd rows
        new_order.append(num_rows - i - 1)  # Add the next row from the bottom

kidneycsv_sh = kidneycsv_rem_nan.iloc[new_order].copy()

kidneycsv_sh.reset_index(drop=True, inplace=True)

print(kidneycsv_sh.head())

"""#Data Splitting"""

from sklearn.model_selection import train_test_split
x = kidneycsv_sh.drop(columns=["classification"])
y = kidneycsv_sh["classification"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0, stratify=y)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""#Data Scaling

"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

'''from sklearn.preprocessing import StandardScaler'''

"""#Knn implementation"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(x_train,y_train)

y_pred = knn_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

"""#Naive_Bayes implementation"""

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train,y_train)
y2_pred = nb_model.predict(x_test)

accuracy = accuracy_score(y_test, y2_pred)
conf_matrix = confusion_matrix(y_test, y2_pred)
report = classification_report(y_test, y2_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

"""#Logistic regression implementation"""

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train,y_train)
y3_pred = lg_model.predict(x_test)

accuracy = accuracy_score(y_test, y3_pred)
conf_matrix = confusion_matrix(y_test, y3_pred)
report = classification_report(y_test, y3_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
