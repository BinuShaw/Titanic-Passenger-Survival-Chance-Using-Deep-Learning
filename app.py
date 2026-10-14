import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title('Passenger Survival Chance In The Titanic Journey')

Pclass = st.slider('Enter the Passenger class for the user', 1,3)
Sex = st.selectbox('Enter the Passenger Gender',['male', 'female'])
SibSp = st.slider('Enter the Passenger total number of Sibling and Spouse', 1,8)
Parch = st.slider('Enter the Passenger total number of Parents and child', 0,6)
Fare = st.number_input('Enter the Fare of the Passenger')
Embarked = st.selectbox('Enter the Passenger Station where they started the journey', ['Southhampton', 'Chebourg', 'Queenstown'])

data = pd.DataFrame(
    {'Pclass' : [Pclass], 
     'Sex' : [Sex],
     'SibSp' : [SibSp],
     'Parch' : [Parch],
     'Fare' : [Fare],
     'Embarked' : [Embarked]
     }
)
# if st.button('Show Passenger Data'):
#     st.write(data)
    

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)
    
with open('onehot_encoder.pkl', 'rb') as file:
    one_hot = pickle.load(file)
    

with open('scaler.pkl', 'rb') as file:
    scale = pickle.load(file)
    

model = load_model('model.h5')

data['Sex']=label.transform(data['Sex'])
embarked = one_hot.transform(data[['Embarked']])

embarked = pd.DataFrame(embarked, columns=one_hot.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']), embarked], axis=1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scale.transform(data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(data)

y = y[0][0]

def chance(y):
    if y>0.5:
        return 'The Passenger will likely to survive the titanic journey'
    else:
        return 'The Passenger will not survive the titanic journey'

if st.button('Predict Survival Chance'):
    st.write('Probability of Passenger Survival Chance',y)
    st.write(chance(y))