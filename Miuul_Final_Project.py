
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import mlxtend
import warnings
import streamlit as st 
import streamlit.components.v1 as components

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from lazypredict.Supervised import LazyClassifier

col1, col2, col3, col4 = st.columns(4,gap = 'Large')

with col1:
   
    st.markdown("<h1 style='text-align: Left; color: green;'>Miuul DA Bootcamp Diabetes Prediction App</h1>", unsafe_allow_html=True)
    st.image('data.jpg', caption='Miuul Data Analyst Bootcamp', width=350)



df = pd.read_csv("diabetes.csv")

df2 = pd.read_csv("death.csv")



Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1


df= df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)



X = df.drop(columns='Outcome')
y = df['Outcome']


scaler = StandardScaler()
X =  pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =1)

models = []

models.append(('KNN Model Accuracy Score', KNeighborsClassifier()))
models.append(('SVC Model Accuracy Score', SVC(gamma='scale')))
models.append(('LR Model Accuracy Score', LogisticRegression(solver='lbfgs', max_iter=4000)))
models.append(('DT Model Accuracy Score', DecisionTreeClassifier()))
models.append(('GNB Model Accuracy Score', GaussianNB()))
models.append(('RF Model Accuracy Score', RandomForestClassifier(n_estimators=100)))
models.append(('GB Model Accuracy Score', GradientBoostingClassifier()))
models.append(('ETC Model Accuracy Score', ExtraTreesClassifier()))


names = []
scores = []

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

# model = RandomForestClassifier().fit(x_train, y_train)
# y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print ("Accuracy Score: %s" % "{0:.2%}".format(accuracy))
print ("Precision Score: %s" % "{0:.2%}".format(precision))
print ("Recall Score: %s" % "{0:.2%}".format(recall))
print ("F1 Score: %s" % "{0:.2%}".format(f1))

with col2:
    name = st.text_input('What is your name?').capitalize()
     
    
    def get_user_input():
         pregnancies = st.number_input("Enter Pregnancies")
         glucose = st.number_input('Enter Glucose')
         bldp = st.number_input("Enter BloodPresssure")
         skin_thickness = st.number_input('Enter Skin Thickness')
         insulin = st.number_input('Enter Insulin')
         BMI = st.number_input('Enter BMI')
         DPF = st.number_input('Enter DPF')
         age = st.number_input('Enter Age')
         
         user_data = {'Pregnancies':pregnancies,
                     'Glucose': glucose,
                     'BloodPressure':bldp,
                     'SkinThickness': skin_thickness,
                     'Insulin': insulin,
                     'BMI': BMI,
                     'DiabetesPedigreeFunction':DPF,
                     'Age': age
                      }
         features = pd.DataFrame(user_data, index=[0])
         return features
    user_input = get_user_input()

with col3:
    
    bt = st.button('Get Result')
    
    if bt:
        
        try:     
            prediction = model.predict(user_input)
            st.session_state['prediction'] = prediction
            st.session_state['name'] = name
            
            st.write('X Train',x_train.shape,'Y Train',y_train.shape )
            st.write('X Test',x_test.shape, 'Y Test',y_test.shape)
            st.write("Confusion Matrix",confusion_matrix(y_test,y_pred))
            st.write(tr_split)
            # st.write("Classification Report", classification_report(y_test,y_pred))
            st.write ("Accuracy Score: %s" % "{0:.2%}".format(accuracy))
            st.write ("Precision Score: %s" % "{0:.2%}".format(precision))
            st.write ("Recall Score: %s" % "{0:.2%}".format(recall))
            st.write ("F1 Score: %s" % "{0:.2%}".format(f1))
            
            if prediction == 1:
                st.session_state['diabetes'] = True
                color = 'red'
            
            else:
                st.session_state['diabetes'] = False
                color = 'green'
                
        except:
            st.warning("Please fill all the required information.")
    
    if 'diabetes' in st.session_state:
        if st.session_state['diabetes']:
            st.write(st.session_state['name'], ":red[You have diabetes. You must eat less :)]")
            
          
            
            st.write("Now select your country and year, see the death rate if you had this disease in this country and in these years")
            country = st.selectbox('Select your country:', df2['Entity'].unique(), key='country')
            year = st.selectbox('Select year:', df2['Year'].unique(), key = 'year')
            
            if country and year:
                
                death_rate = df2[(df2['Entity'] == country) & (df2['Year'] == year)]['Deaths'].iloc[0]
                st.write(f"In :red[{country}], in this year :red[{year}] the average death rate due to diabetes is :red[{death_rate:.2f}%].")
                st.write(":green[Pray you are not sick]")
        else:
            st.write(st.session_state['name'], ":green[You don't have Diabetes. You can eat more :)]")
        
with col4:
    
    prediction = model.predict(user_input)
    if prediction == 1:
        color = 'red'
    else:
        color = 'green'
    
    st.header('Age-Pregnancy')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
    ax2 = sns.scatterplot(x = user_input['Age'], y = user_input['Pregnancies'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(0,20,2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)
    
    st.header('Age-Glucose')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
    ax4 = sns.scatterplot(x = user_input['Age'], y = user_input['Glucose'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(30,220,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)
    
    st.header('Age-Blood Pressure')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
    ax6 = sns.scatterplot(x = user_input['Age'], y = user_input['BloodPressure'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(30,130,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)
    
    st.header('Age-Skin Thickness')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
    ax8 = sns.scatterplot(x = user_input['Age'], y = user_input['SkinThickness'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(10,70,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)
    
    st.header('Age-Insulin')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
    ax10 = sns.scatterplot(x = user_input['Age'], y = user_input['Insulin'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(0,350,50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)
    
    
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
    ax12 = sns.scatterplot(x = user_input['Age'], y = user_input['BMI'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(15,55,5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)
    
    
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
    ax14 = sns.scatterplot(x = user_input['Age'], y = user_input['DiabetesPedigreeFunction'], s = 150, color = color)
    plt.xticks(np.arange(20,80,5))
    plt.yticks(np.arange(0,1.3,0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)