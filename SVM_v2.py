import streamlit as st

st.title("""
     Prediksi Kelayakan Nasabah
    **Pada Bank Perkreditan Rakyat Syariah**
""")

import pandas as pd
import numpy as np



st.sidebar.header('User Input Parameters')

def user_input_features():
    d_1 = ("Belum Kawin", "Janda","Duda","Kawin")
    o_1 = list(range(len(d_1)))
    v_1 = st.sidebar.selectbox("Status Pernikahan", o_1, format_func=lambda x: d_1[x])

    d_2 = ("Laki-Laki", "Perempuan")
    o_2 = list(range(len(d_2)))
    v_2 = st.sidebar.selectbox("Jenis Kelamin", o_2, format_func=lambda x: d_2[x])

    d_3 = ("Ya", "Tidak")
    o_3 = list(range(len(d_3)))
    v_3 = st.sidebar.selectbox("Asuransi", o_3, format_func=lambda x: d_3[x])

    d_4 = ("Wirausaha", "Guru","Karyawan Swasta","Wiraswasta","Swasta","TNI AU","PNS","Polisi","Buruh",
      "Pedagang","Pensiunan")
    o_4 = list(range(len(d_4)))
    v_4 = st.sidebar.selectbox("Pekerjaan", o_4, format_func=lambda x: d_4[x])

    d_5 = ("(750000.0, 3200000.0", "3200000.0, 6000000.0","6000000.0, 10000000.0" ,
      "10000000.0, 15000000.0","15000000.0, 25000000.0" ,"25000000.0, 30000000.0",
      "30000000.0, 40200000.0" ,"40200000.0, 55000000.0","55000000.0, 70000000.0" ,
      "70000000.0, 85000000.0","85000000.0, 106000000.0" ,"106000000.0, 150000000.0",
      "150000000.0, 200000000.0" ,"200000000.0, 250000000.0","250000000.0, 300000000.0")
    o_5 = list(range(len(d_5)))
    v_5 = st.sidebar.selectbox("gaji_pribadi", o_5, format_func=lambda x: d_5[x])

    d_6 = ("499999.999, 1500000.0", "1500000.0, 2000000.0","2000000.0, 2500000.0",
      "2500000.0, 3500000.0","3500000.0, 4500000.0","4500000.0, 10000000.0","10000000.0, 35000000.0",
      "Tidak Berpenghasilan")
    o_6 = list(range(len(d_6)))
    v_6 = st.sidebar.selectbox("gaji_pasangan", o_6, format_func=lambda x: d_6[x])

    d_7 = ("24.0, 28.0", "28.0, 32.0","32.0, 35.0","35.0, 38.0","38.0, 42.0","42.0, 45.0","45.0, 49.0",
      "49.0, 53.0","53.0, 60.0","60.0, 76.0")
    o_7 = list(range(len(d_7)))
    v_7 = st.sidebar.selectbox("usia", o_7, format_func=lambda x: d_7[x])

    d_8 = ("1299999.999, 3000000.0", "3000000.0, 5000000.0","5000000.0, 5030000.0","5030000.0, 8000000.0",
      "8000000.0, 10000000.0","10000000.0, 12725000.0","12725000.0, 15000000.0","15000000.0, 20000000.0",
      "20000000.0, 27000000.0","27000000.0, 39475000.0","39475000.0, 50000000.0","50000000.0, 70000000.0",
      "70000000.0, 100000000.0","100000000.0, 120000000.0","120000000.0, 224700000.0","224700000.0, 350000000.0",
      "350000000.0, 2000000000.0")
    o_8 = list(range(len(d_8)))
    v_8 = st.sidebar.selectbox("jumlah_pembiayaan", o_8, format_func=lambda x: d_8[x])

    d_9 = ("1.999, 3.0", "3.0, 5.02","5.02, 6.0","6.0, 11.05","11.05, 12.0","12.0, 18.0","18.0, 24.0",
      "24.0, 30.0","30.0, 34.0","34.0, 36.0","36.0, 43.54","43.54, 48.0","48.0, 49.0","49.0, 54.0",
      "54.0, 60.0","60.0, 72.0","72.0, 120.0")
    o_9 = list(range(len(d_9)))
    v_9 = st.sidebar.selectbox("jangka_waktu", o_9, format_func=lambda x: d_9[x])

    d_10 = ("0.999, 5.0", "5.0, 8.0","8.0, 10.0","10.0, 13.0","13.0, 15.0","15.0, 16.0","16.0, 20.0",
      "20.0, 23.0","23.0, 28.0","28.0, 44.0")
    o_10 = list(range(len(d_10)))
    v_10 = st.sidebar.selectbox("lama_usaha", o_10, format_func=lambda x: d_10[x])

    d_11 = ("Ada", "Tidak")
    o_11 = list(range(len(d_11)))
    v_11 = st.sidebar.selectbox("pinjaman_ditempat_lain", o_11, format_func=lambda x: d_11[x])

    
    datax = {'status_pernikahan': v_1,
            'jenis_kelamin': v_2,
            'asuransi': v_3,
            'pekerjaan': v_4,
            'gaji_pribadi': v_5,
            'gaji_pasangan': v_6,
            'usia': v_7,
            'jumlah_pembiayaan': v_8,
            'jangka_waktu': v_9,
            'lama_usaha': v_10,
            'pinjaman_ditempat_lain': v_11
            }
    features = pd.DataFrame(datax, index=[0])
    return features


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
df = user_input_features()
  
prediction = best.predict(df)
prediction_proba = best.predict_proba(df)

if st.button('PREDIKSI !!!!!!'):

  def test():
    st.info("Lancar")
  def test2():
    st.info("Kurang Lancar")

  st.subheader('Hasil Prediksi')
  if prediction == 1:
      test()
  else:
      test2()

  st.subheader('Probabilitas Prediksi')
  st.write(prediction_proba)

  st.subheader('Keterangan index dan label')
  st.write(pd.DataFrame({
    'Label': ['Kurang Lancar','Lancar']}))





