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
# Display the DataFram
st.write(df.head())
st.write("Shape:", df.shape)


# data splitting and processing

# Add categories
df['height'] = df['height'] * 0.01
df['bmi'] = (df['weight'] / (df['height'] ** 2)).astype('int')

# Define BMI categories


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


# Apply the function to create a new feature
df['bmi_category'] = df['bmi'].apply(categorize_bmi)

# Define age groups


def categorize_age(age):
    if age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'


# Apply the function to create a new feature
df['age_group'] = df['age'].apply(categorize_age)

# Define blood pressure categories


def categorize_bp(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif ap_hi >= 140 or ap_lo >= 90:
        return 'Hypertension'
    else:
        return 'High-Normal'


# Apply the function to create a new feature
df['bp_category'] = df.apply(lambda row: categorize_bp(
    row['ap_hi'], row['ap_lo']), axis=1)

# Display the first few rows of the DataFrame
st.write("First few rows of the DataFrame with BMI, Age Group, and Blood Pressure categories:")
st.write(df.head())

# Display a bar chart showing the distribution of BMI categories
st.write("Distribution of BMI categories:")
st.bar_chart(df['bmi_category'].value_counts())

# Display a bar chart showing the distribution of Age groups
st.write("Distribution of Age groups:")
st.bar_chart(df['age_group'].value_counts())

# Display a bar chart showing the distribution of Blood Pressure categories
st.write("Distribution of Blood Pressure categories:")
st.bar_chart(df['bp_category'].value_counts())

# Class Mapping
st.write("Class Mapping")
age_group_map = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
# Define mapping dictionaries
bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
st.write("bmi_map:", bmi_mapping)
age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
st.write("age_group_map:", age_group_map)
bp_mapping = {'Normal': 0, 'High-Normal': 1, 'Hypertension': 2}
st.write("bp_map:", bp_mapping)

# Map values in the category columns using the mapping dictionaries
df['bmi_category'] = df['bmi_category'].map(bmi_mapping)
df['age_group'] = df['age_group'].map(age_mapping)
df['bp_category'] = df['bp_category'].map(bp_mapping)

st.write("First few rows of the DataFrame after mapping:")
st.write(df.head())

# generate 2d classification dataset
st.write("Data splitting and processing")
st.write("X = df.drop('cardio', axis=1).to_numpy()")
X = df.drop('cardio', axis=1).to_numpy()

st.write("X[:2]", X[:2])
st.write("X.shape:", X.shape)

st.write("y = df['cardio'].to_numpy()")
y = df['cardio'].to_numpy()
st.write("y[:2]", y[:2])
st.write('y.shape:', y.shape)

# data Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
st.write("X after scaling")
st.write("X[:2]", X[:2])

# Data Splitting
from sklearn.model_selection import train_test_split
st.write("Data Splitting using train_test_split")
st.write("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write("X_train.shape:", X_train.shape)
st.write("X_test.shape:", X_test.shape)
st.write("y_train.shape:", y_train.shape)
st.write("y_test.shape:", y_test.shape)

st.write("Further Splitting the training data into training and validation sets ")
st.write("x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)")
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
st.write("x_train.shape:", x_train.shape)
st.write("x_val.shape:", x_val.shape)
st.write("y_train.shape:", y_train.shape)
st.write("y_val.shape:", y_val.shape)


# Model Building
st.write("Model Building")


# Define the model
model = Sequential([
    Dense(500, input_dim=x_train.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Implement a learning rate scheduler


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)


callback = LearningRateScheduler(scheduler)

# Compile the model with an optimizer and learning rate
model.compile(loss=binary_crossentropy,
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])


# Display model summary using Streamlit

# Redirect stdout to capture the model summary
buffer = io.StringIO()
with redirect_stdout(buffer):
    model.summary()

# Get the captured model summary
model_summary_text = buffer.getvalue()

# Display the model summary in Streamlit
st.text("Model Summary:")
st.text(model_summary_text)

# Train the model
# st.write("Training the model")
# history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, callbacks=[callback])
# st.write("Model Training Completed")


# # Evaluate the model
# train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# # Print the evaluation results
# st.write("Evaluation Results:")
# st.write(f"Train Accuracy: {train_acc:.3f}")
# st.write(f"Test Accuracy: {test_acc:.3f}")

# # Plot training history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# st.pyplot(plt)

# # evaluate the model
# st.write("Evaluating the model")
# st.write("score = model.evaluate(X_test, y_test, verbose=0)")
# score = model.evaluate(X_test, y_test, verbose=0)
# st.write("Test loss:", score[0])
# st.write("Test accuracy:", score[1])

# Button to initiate training
if st.button("Train Model"):
    st.write("Training the model...")

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(
        x_val, y_val), epochs=20, batch_size=64, callbacks=[callback])

    st.write("Model Training Completed")

    # Display evaluation results
    st.subheader("Evaluation Results")

    # Evaluate the model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # st.write(f"Train Accuracy: {train_acc:.3f}")
    # st.write(f"Test Accuracy: {test_acc:.3f}")

    # Plot training history
    st.subheader("Training History")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

    # Evaluate the model on test data
    st.subheader("Model Evaluation on Test Data")
    st.write("Evaluating the model...")
    score = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test loss: {score[0]:.2f}")
    st.write(f"Test accuracy: {score[1]:.2f}")
    # st.write(f"Test loss: {score[0]:.2f}")
    # st.write(f"Test accuracy: {score[1]:.2f}")
