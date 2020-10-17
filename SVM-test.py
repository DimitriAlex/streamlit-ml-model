import streamlit as st

st.title("""
     Prediksi Kelayakan Nasabah
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
    status_pernikahan = st.sidebar.slider('status_pernikahan', 0, 3, 1)
    jenis_kelamin = st.sidebar.slider('jenis_kelamin', 0, 1, 1)
    asuransi = st.sidebar.slider('asuransi', 0, 1, 1)
    pekerjaan = st.sidebar.slider('pekerjaan', 0, 10, 1)
    gaji_pribadi = st.sidebar.slider('gaji_pribadi', 0, 14, 1)
    gaji_pasangan = st.sidebar.slider('gaji_pasangan', 0, 7, 1)
    usia = st.sidebar.slider('usia', 0, 9, 1)
    jumlah_pembiayaan = st.sidebar.slider('jumlah_pembiayaan', 0, 15, 1)
    jangka_waktu = st.sidebar.slider('jangka_waktu', 0, 16, 1)
    lama_usaha = st.sidebar.slider('lama_usaha', 0, 9, 1)
    pinjaman_ditempat_lain = st.sidebar.slider('pinjaman_ditempat_lain', 0, 1, 1)

    datax = {'status_pernikahan': status_pernikahan,
            'jenis_kelamin': jenis_kelamin,
            'asuransi (ya atau tidak)': asuransi,
            'pekerjaan': pekerjaan,
            'gaji_pribadi': gaji_pribadi,
            'gaji_pasangan': gaji_pasangan,
            'usia': usia,
            'jumlah_pembiayaan': jumlah_pembiayaan,
            'jangka_waktu': jangka_waktu,
            'lama_usaha': lama_usaha,
            'pinjaman_ditempat_lain': pinjaman_ditempat_lain}
    features = pd.DataFrame(datax, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

prediction = best.predict(df)
prediction_proba = best.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame({
  'Label': ['Kurang Lancar','Lancar']}))

def test():
    st.info("Lancar")

def test2():
    st.info("Kurang Lancar")


st.subheader('Prediction')
if prediction == 1:
    test()
else:
    test2()

st.subheader('Prediction Probability')
st.write(prediction_proba)