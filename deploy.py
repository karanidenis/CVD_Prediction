import io
from contextlib import redirect_stdout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# Title container
with st.container():
    st.title("Machine Learning Model to test for cardio Vascular Diseases")
    st.subheader("This is the data set for the model")

df = pd.read_csv('./cardio_train.csv', sep=';')

# Display the DataFrame
st.write(df.head())

# drop Id and change age to years
df.drop('id', axis=1, inplace=True)
df['age'] = (df['age'] / 365).round().astype('int')
# add subheader to explain the data
st.subheader("DataFrame after dropping Id and changing age to years")
# Display the DataFrame
st.write(df.head())
st.write("Shape:", df.shape)

# Add categories
df['height'] = df['height'] * 0.01
df['bmi'] = (df['weight'] / (df['height'] ** 2)).astype('int')


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


df['bmi_category'] = df['bmi'].apply(categorize_bmi)


def categorize_age(age):
    if age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'


df['age_group'] = df['age'].apply(categorize_age)


def categorize_bp(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif ap_hi >= 140 or ap_lo >= 90:
        return 'Hypertension'
    else:
        return 'High-Normal'


df['bp_category'] = df.apply(lambda row: categorize_bp(
    row['ap_hi'], row['ap_lo']), axis=1)

st.write("First few rows of the DataFrame with BMI, Age Group, and Blood Pressure categories:")
st.write(df.head())

st.write("Distribution of BMI categories:")
st.bar_chart(df['bmi_category'].value_counts())

st.write("Distribution of Age groups:")
st.bar_chart(df['age_group'].value_counts())

st.write("Distribution of Blood Pressure categories:")
st.bar_chart(df['bp_category'].value_counts())

st.write("Class Mapping")
age_group_map = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
bp_mapping = {'Normal': 0, 'High-Normal': 1, 'Hypertension': 2}

df['bmi_category'] = df['bmi_category'].map(bmi_mapping)
df['age_group'] = df['age_group'].map(age_mapping)
df['bp_category'] = df['bp_category'].map(bp_mapping)

st.write("First few rows of the DataFrame after mapping:")
st.write(df.head())

X = df.drop('cardio', axis=1).to_numpy()
y = df['cardio'].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

st.write("Data Splitting and Processing Completed")

# Model Building
st.write("Model Building")


def build_model():
    model = Sequential([
        Dense(500, input_dim=x_train.shape[1], activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss=binary_crossentropy,
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model


# Function to display model summary
def display_model_summary(model):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()
    model_summary_text = buffer.getvalue()
    st.text("Model Summary:")
    st.text(model_summary_text)


model = build_model()
display_model_summary(model)

# Button to initiate training
if st.button("Train Model"):
    st.write("Training the model...")
    history = model.fit(x_train, y_train, validation_data=(
        x_val, y_val), epochs=20, batch_size=64)
    st.write("Model Training Completed")
    st.text("Evaluating the model on test data...")
    score = model.evaluate(X_test, y_test, verbose=0)
    st.write("Test loss:", score[0])
    st.write("Test accuracy:", score[1])

# Check if the file is uploaded
uploaded_file = st.file_uploader("Upload new data", type=["csv"])

if uploaded_file is not None:
    st.write("Processing new data...")

    # Read the uploaded file
    new_data = pd.read_csv(uploaded_file, sep=';')

    # Drop ID and convert age to years
    new_data.drop('id', axis=1, inplace=True)
    new_data['age'] = (new_data['age'] / 365).round().astype('int')

    # Add categories
    new_data['height'] = new_data['height'] * 0.01
    new_data['bmi'] = (new_data['weight'] /
                       (new_data['height'] ** 2)).astype('int')

    new_data['bmi_category'] = new_data['bmi'].apply(categorize_bmi)
    new_data['age_group'] = new_data['age'].apply(categorize_age)
    new_data['bp_category'] = new_data.apply(lambda row: categorize_bp(
        row['ap_hi'], row['ap_lo']), axis=1)

    new_data['bmi_category'] = new_data['bmi_category'].map(bmi_mapping)
    new_data['age_group'] = new_data['age_group'].map(age_mapping)
    new_data['bp_category'] = new_data['bp_category'].map(bp_mapping)

    new_X = new_data.drop
