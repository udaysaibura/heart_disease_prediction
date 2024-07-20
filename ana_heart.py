pip install seaborn
pip install matplotlib
pip install scikit-learn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv(r"C:\Users\udays\OneDrive\Desktop\GRP_PRJ\framingham.csv")
df.head()
#df.shape
df.describe()
df.keys()
df.info()
df.dropna(axis=0,inplace=True)
#df.shape
df.isna().sum()
plt.boxplot(df.male)
plt.boxplot(df.age)
df.drop("education",axis=1,inplace=True)
df.describe()
df.head()
df["TenYearCHD"].value_counts()
df.describe()
df.head()
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='Purples', annot=True, linecolor='Green', linewidths=1.0)
plt.show()
# sns.catplot(data=df, kind='count', x='male',hue='currentSmoker')
plt.show()
sns.catplot(data=df, kind='count', x='TenYearCHD', col='male',row='currentSmoker', palette='Blues')
plt.show()
# from sklearn.preprocessing import MinMaxScaler
# standardScaler = StandardScaler()
columns_to_scale = [ 'cigsPerDay', 'totChol','sysBP','diaBP','BMI','heartRate', 'glucose']
# df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
# df['age']=df['age']/100;
cigsperday=[]
cigsperday.append(min(df['cigsPerDay']))
cigsperday.append(max(df['cigsPerDay']))
totchol=[]
totchol.append(min(df['totChol']))
totchol.append(max(df['totChol']))
sysbp=[]
sysbp.append(min(df['sysBP']))
sysbp.append(max(df['sysBP']))
diabp=[]
diabp.append(min(df['diaBP']))
diabp.append(max(df['diaBP']))
bmi=[]
bmi.append(min(df['BMI']))
bmi.append(max(df['BMI']))
heartrate=[]
heartrate.append(min(df['heartRate']))
heartrate.append(max(df['heartRate']))
glu=[]
glu.append(min(df['glucose']))
glu.append(max(df['glucose']))
from sklearn.preprocessing import MinMaxScaler
minmaxscale=MinMaxScaler()
df[columns_to_scale] = minmaxscale.fit_transform(df[columns_to_scale])
df['age']=df['age']/100;
x=df.iloc[:,0:14]
y=df.iloc[:,14:15]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=11)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
log_y_pred=lr.predict(X_test)
#score=lr.score(X_test,y_test)

sc=accuracy_score(log_y_pred,y_test)
#sc
#score
#new data male,age,education,currentsmoker cigsPerDay BPMeds prevalentStroke diabetes totChol  sysBP diaBP BMI heartRate glucose         
new_data1=[[1,0.45,1,0.9,1,1,1,1,2.20,0.90,0.90,0.27,0.80,0.80]]
if lr.predict(new_data1)==1:
    print("Chance of heart Attack")
else:
    print("Ur safe ")
from sklearn.neighbors import KNeighborsClassifier
df.head()
knn_scores = []
for k in range(3,21,2):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
#knn_scores
def calculate_knn_scores(X, y, k_range):
    knn_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        knn_scores.append(score)
    return knn_scores

def plot_elbow_method(k_range, knn_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, knn_scores, marker='o', linestyle='-', color='blue')
    for i in range(len(k_range)):
        plt.text(k_range[i], knn_scores[i], f'({k_range[i]}, {knn_scores[i]:.2f})', fontsize=9, verticalalignment='bottom')
    
    plt.xticks(k_range)
    plt.xlabel('Number of Neighbors (K)', color='red', weight='bold', fontsize=12)
    plt.ylabel('Scores', color='red', weight='bold', fontsize=12)
    plt.title('K Neighbors Classifier Scores for Different K Values', color='red', weight='bold', fontsize=12)
    plt.grid(True)
    plt.show()
X_train2, X_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=42)

# Define the range of k values to evaluate
k_range = range(1, 50)

# Calculate KNN scores for each k value
knn_scores = calculate_knn_scores(X_train2, y_train2, k_range)

# Plot the Elbow Method graph
plot_elbow_method(k_range, knn_scores)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
knn_y_pred=(knn.predict(X_test))
#score=accuracy_score(knn_y_pred,y_test)
#score
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=3)
model.fit(X_train, y_train)

ran_y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,ran_y_pred)
sns.heatmap(cm/np.sum(cm),annot=True,fmt=".2%",cmap='Blues')
plt.show()
ac = accuracy_score(y_test,ran_y_pred)
#print('Accuracy is: ',ac*100)
res=[]
for i in range(len(y_test)):
    z=0
    o=0
    if log_y_pred[i]==1:
        o+=1
    else:
        z+=1
    if knn_y_pred[i]==1:
        o+=1
    else:
        z+=1
    if ran_y_pred[i]==1:
        o+=1
    else:
        z+=1
    if z>o:
        res.append(0)
    else:
        res.append(1)
score=accuracy_score(res,y_test)
#score
def stan(data):
    # print(data)
    data[:, 0] = data[:, 1] / 100
    data[:, 3] = (data[:, 3] - cigsperday[0]) / (cigsperday[1] - cigsperday[0])
    data[:, 8] = (data[:, 8] - totchol[0]) / (totchol[1] - totchol[0])
    data[:, 9] = (data[:, 9] - sysbp[0]) / (sysbp[1] - sysbp[0])
    data[:, 10] = (data[:, 10] - diabp[0]) / (diabp[1] - diabp[0])
    data[:, 11] = (data[:, 11] - bmi[0]) / (bmi[1] - bmi[0])
    data[:, 12] = (data[:, 12] - heartrate[0]) / (heartrate[1] - heartrate[0])
    data[:, 13] = (data[:, 13] - glu[0]) / (glu[1] - glu[0])
    return data
def ret(point):
    if (lr.predict(point)==1 and knn.predict(point)==1):
        return 1
    if (lr.predict(point)==1 and model.predict(point)==1):
        return 1
    if (model.predict(point)==1 and knn.predict(point)==1):
        return 1
    return 0
def predict(data):
    if ret(data)==1:
        print("Waring::Chance of heart Attack")
    else:
        print("Ur safe ")

# if ret(stan(new_data3))==1:
#     print("Waring:chance of heart Attack")
# else:
#     print("Ur safe ")
new_data1=np.array([[1,25,0,0,1,1,1,0,120,100,80,18,60,60]])
new_data2=np.array([[1,10,0,0,0,0,0,0,100,120,80,23,80,70]])
new_data3 = np.array([[1, 45, 1, 1, 1, 1, 1, 1, 180, 160, 120, 23, 80, 70]])
new_data4=np.array([[1,25,0,0,1,1,1,1,120,100,80,18,60,60]])
#predict(stan(new_data4))
#predict(stan(new_data2))
#predict(stan(new_data3))
#predict(stan(new_data1))
import streamlit as st
import numpy as np

# Assuming `stan` and `predict` functions are already defined
def stan(data):
    # Replace placeholders with actual min and max values
    cigsperday = (0, 100)
    totchol = (100, 300)
    sysbp = (90, 180)
    diabp = (60, 120)
    bmi = (18, 40)
    heartrate = (60, 100)
    glu = (70, 140)

    data[:, 0] = data[:, 1] / 100
    data[:, 3] = (data[:, 3] - cigsperday[0]) / (cigsperday[1] - cigsperday[0])
    data[:, 8] = (data[:, 8] - totchol[0]) / (totchol[1] - totchol[0])
    data[:, 9] = (data[:, 9] - sysbp[0]) / (sysbp[1] - sysbp[0])
    data[:, 10] = (data[:, 10] - diabp[0]) / (diabp[1] - diabp[0])
    data[:, 11] = (data[:, 11] - bmi[0]) / (bmi[1] - bmi[0])
    data[:, 12] = (data[:, 12] - heartrate[0]) / (heartrate[1] - heartrate[0])
    data[:, 13] = (data[:, 13] - glu[0]) / (glu[1] - glu[0])
    return data

def predict(data):
    # Placeholder for your predict function
    return ret(data)

def ret(point):
    if (lr.predict(point) == 1 and knn.predict(point) == 1):
        return 1
    if (lr.predict(point) == 1 and model.predict(point) == 1):
        return 1
    if (model.predict(point) == 1 and knn.predict(point) == 1):
        return 1
    return 0

st.title("Heart Attack Prediction")

# Input fields
male=st.number_input("Male 1  Female-0", min_value=0, max_value=1, value=0)
age = st.number_input("Age", value=25)
cigs_per_day = st.number_input("Cigs Per Day", value=0)
bp_meds = st.number_input("BP Meds (0 or 1)", min_value=0, max_value=1, value=0)
prevalent_stroke = st.number_input("Prevalent Stroke (0 or 1)", min_value=0, max_value=1, value=0)
prevalent_hyp = st.number_input("Prevalent Hyp (0 or 1)", min_value=0, max_value=1, value=1)
diabetes = st.number_input("Diabetes (0 or 1)", min_value=0, max_value=1, value=0)
tot_chol = st.number_input("Total Cholesterol", value=180)
sys_bp = st.number_input("Systolic BP",value=120)
dia_bp = st.number_input("Diastolic BP", value=80)
bmi = st.number_input("BMI", value=23.0)
heart_rate = st.number_input("Heart Rate",  value=70)
glu = st.number_input("Glucose",  value=80)

# Create the input array
new_data = np.array([[1, age, cigs_per_day, bp_meds, prevalent_stroke, prevalent_hyp, diabetes, 1, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glu]])

# Standardize the input data
stan_data = stan(new_data)

# Get the prediction
if st.button("Predict"):
    result = predict(stan_data)
    if result== 1:
        st.error("🚨 Emergency: High chance of heart attack! 🚨")
    else:
        st.success("You are safe.")

# Run the app using streamlit run app.py
