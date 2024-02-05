import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
# Set Streamlit app style
st.markdown(
    """
    <style>
        body {
            background-color: #f0f0f0;  /* Light gray background */
        }
        .css-1bxjvdi {
            color: #3366cc;  /* Dark blue text */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# load data
data = pd.read_csv('cleaned data.csv')


# separate legitimate and fraudulent transactions
normal = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
normal_sample = normal.sample(n=len(fraud), random_state=2)
data = pd.concat([normal_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")


# create input fields for user to enter feature values
input_df = st.text_input('Input the features')

input_df_lst = input_df.split(',')
# Create a button to submit input and get prediction





predict= st.button("Predict")

if predict:

    features = np.array(input_df_lst, dtype=np.float64)
    # Make prediction
    prediction = model.predict(features.reshape(1,-1))
    
    # Display result
    if prediction[0] == 0:
        st.write("Normal transaction")
    else:
        st.write("Fraudulent transaction")
elif predict:
    st.write("Invalid input. Please provide values for all features.")
