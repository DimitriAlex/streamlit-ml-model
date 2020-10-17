import streamlit as st

st.title("""
    # Prediksi Kelayakan Nasabah
    **Pada Bank Perkreditan Rakyat Syariah**
""")

import pandas as pd
import numpy as np

data = pd.read_excel('output.xlsx')
data = data.drop(['Unnamed: 0'], axis=1)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

# Train 70% test 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.2)

from sklearn.pipeline import make_pipeline
from sklearn import svm

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score, roc_auc_score

pipeline = make_pipeline(svm.SVC())
pipeline.get_params()
parameters = {'svc__kernel': ['rbf'], 'svc__C': [1,10],'svc__probability': [True],'svc__random_state': [42],
              'svc__class_weight':['balanced'], 'svc__gamma':[0.1,1]}

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, shuffle=True)
clf = GridSearchCV(pipeline, parameters, cv=cv, scoring=['roc_auc','recall','accuracy'],refit=False)
clf.fit(X_train, y_train)

model = clf.cv_results_['params'][2]

best = svm.SVC(C=model['svc__C'],
               class_weight=model['svc__class_weight'],
               gamma =model['svc__gamma'],
               kernel=model['svc__kernel'],
               probability=model['svc__probability'],
               random_state=model['svc__random_state'])

y_pred = best.fit(X_train, y_train).predict(X_test)

names = ['Kurang Lancar','Lancar']


st.sidebar.header('User Input Parameters')

def user_input_features():
    status_pernikahan = st.sidebar.selectbox("Status Pernikahan",("Sudah Menikah","Belum Menikah","Janda","Duda"))
    jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin",(""))
    asuransi = st.sidebar.selectbox("Asuransi",(""))
    pekerjaan = st.sidebar.selectbox("Pekerjaan",(""))
    gaji_pribadi = st.sidebar.selectbox("Penghasilan Pribadi",(""))
    gaji_pasangan = st.sidebar.selectbox("Penghasilan Pasangan",(""))
    usia = st.sidebar.selectbox("Usia",(""))
    jumlah_pembiayaan = st.sidebar.selectbox("Jumlah Peminjamaan",(""))
    jangka_waktu = st.sidebar.selectbox("Jangka Waktu Pembayaran",(""))
    lama_usaha = st.sidebar.selectbox("Lama Usaha",(""))
    pinjaman_ditempat_lain == st.sidebar.selectbox("Pinjaman di Tempat Lain",(""))

    datax = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(datax, index=[0])
    return features



auc = roc_auc_score(y_test, y_pred)
st.subheader("Nilai AUC")
st.write(auc)
