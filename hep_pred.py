import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

columNames = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG',
              'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT',
              'ALBUMIN', 'PROTIME', 'HISTOLOGY']
df = pd.read_csv("hepatitis.data", names=columNames)
df.replace("?", np.nan, inplace=True)
df.isnull().sum()

# Convert the type of numericals
numerical_variables = ['AGE', 'BILIRUBIN', 'PROTIME', 'ALBUMIN', 'ALK PHOSPHATE', 'SGOT']
df["BILIRUBIN"] = df.BILIRUBIN.astype(float)
df["PROTIME"] = df.PROTIME.astype(float)
df["ALK PHOSPHATE"] = df["ALK PHOSPHATE"].astype(float)
df["SGOT"] = df.SGOT.astype(float)
df["ALBUMIN"] = df.ALBUMIN.astype(float)

# empty space to mean
df[numerical_variables].fillna(df[numerical_variables].mean()).head(5)

# Categorical variables
categorical_variables = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
                         'SPLEEN PALPABLE',
                         'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']

# Fmissing data -> most frequent data
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_mean.fit(df)
imputed_train_df = imp_mean.transform(df)
imputedDf = pd.DataFrame(imputed_train_df, columns=columNames)

# normalizing
sc = StandardScaler()
sc.fit(imputedDf.drop(["Class", "PROTIME", "BILIRUBIN"], axis=1))
scaled_features = sc.transform(imputedDf.drop(["Class", "PROTIME", "BILIRUBIN"], axis=1))
X = scaled_features
y = imputedDf["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
sm = SMOTE(random_state=33)
X_train_new, y_train_new = sm.fit_resample(X_train, y_train.ravel())

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_new, y_train_new)

pickle.dump(knn, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
