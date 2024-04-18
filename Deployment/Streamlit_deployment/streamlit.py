import streamlit as st
import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import LearningRateScheduler
from contextlib import redirect_stdout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import joblib
from keras.models import model_from_json


df = pd.read_csv(
    '../cardio_train.csv', sep=';')
# df.drop('id', axis=1, inplace=True)

def analysis_visualizations():
    with st.container():
        st.header("Data analysis and Visualization")
        st.subheader("This is the data set for the model")
    
    st.sidebar.subheader("Display Data")
    rows_to_display = st.sidebar.slider(
        "Number of Rows", min_value=1, max_value=100, value=5)
    st.write(df.head(rows_to_display))
    st.write("Shape:", df.shape)


    with st.container():
        # st.title("Exploratory Data Analysis - Numeric Features")
        st.subheader("Correlation Matrix of Numeric Features (Heatmap)")
    features = df.columns
    # Create a subset of the DataFrame with only numeric features
    numeric_df = df[features]
    # print(df)
    numeric_df = df

    # Correlation matrix to see how features correlate with cardio
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    
    # Define the list of features for visualization
    features = ['age', 'weight', 'height', 'cholesterol', 'gender', 'gluc', 'active', 'smoke', 'alco', 'cardio', 'ap_hi', 'ap_lo']
    df['age'] = (df['age'] / 365).round().astype('int')

    # Sidebar
    selected_feature = st.sidebar.selectbox("Select Feature to see Visualization", features)

    # Title container
    with st.container():
        st.title(f"Visualization - {selected_feature} Distribution")

    # # Check if the selected feature is 'age', convert it to years if so
    # if selected_feature == 'age':
    #     df[selected_feature] = (df[selected_feature] / 365).round().astype('int')

    # Visualize the distribution of the selected feature
    feature_counts = df[selected_feature].value_counts()
    st.bar_chart(feature_counts)



df = pd.read_csv('../cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)

def data_preprocessing():
    st.header("Data Preprocessing")
    # drop Id and change age to years
    df['age'] = (df['age'] / 365).round().astype('int')
    df['height'] = df['height'] * 0.01

    # add subheader to explain the data
    st.subheader(
        "DataFrame after dropping Id and changing age to years and height to meters:")
    # Display the DataFrame
    st.write(df.head())
    st.write("Shape:", df.shape)

    # Add categories
    st.subheader(
        "Add rows to the DataFrame; BMI, Age Group, and Blood Pressure categories:")
    st.write("Calculate BMI and add to the DataFrame: Bmi = weight / (height ** 2)")
    df['bmi'] = (df['weight'] / (df['height'] ** 2)).astype('int')

    # Display the explanation to the user
    st.write("Explanation:")
    st.write(
        "We have added three new categories to the DataFrame based on existing columns:")
    st.write(
        "- BMI Category: Categorized into Underweight: 0, Normal: 1, Overweight: 2, and Obese: 3")
    st.write("- Age Group: Categorized into Young: 0, Middle-aged: 1, and Senior: 2")
    st.write("- Blood Pressure Category: Categorized into Normal: 0, Hypertension: 1, and High-Normal: 2")

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
    bmi_mappping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_mappping)

    def categorize_age(age):
        if age < 40:
            return 'Young'
        elif 40 <= age < 60:
            return 'Middle-aged'
        else:
            return 'Senior'

    df['age_group'] = df['age'].apply(categorize_age)
    age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
    df['age_group'] = df['age_group'].map(age_mapping)

    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 'Normal'
        elif ap_hi >= 140 or ap_lo >= 90:
            return 'Hypertension'
        else:
            return 'High-Normal'

    df['bp_category'] = df.apply(lambda row: categorize_bp(
        row['ap_hi'], row['ap_lo']), axis=1)
    bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
    df['bp_category'] = df['bp_category'].map(bp_mapping)

    # Display the DataFrame
    st.write("Updated DataFrame:")
    st.write(df.head())

    # Define the list of features for visualization
    features = ['bmi', 'bmi_category', 'age_group',
                    'bp_category', 'cardio']

    # Sidebar
    selected_feature = st.sidebar.selectbox(
        "Select Feature to see Visualization of new features", features)

    # Title container
    with st.container():
        st.title(f"Visualization - {selected_feature} Distribution")

    # Visualize the distribution of the selected feature
    feature_counts = df[selected_feature].value_counts()
    st.bar_chart(feature_counts)
    
    with st.container():
        # st.title("Exploratory Data Analysis - Numeric Features")
        st.subheader("Correlation Matrix of new Numeric Features (Heatmap)")
    # features = ['bmi', 'bmi_category', 'age_group', 'bp_category', 'cardio', 'ap_hi', 'ap_lo']
    # Create a subset of the DataFrame with only numeric features
    numeric_df = df[features]
    # print(df)
    print(numeric_df)
    # numeric_df = df

    # Correlation matrix to see how features correlate with cardio
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    


df = pd.read_csv('../cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)
df['age'] = (df['age'] / 365).round().astype('int')
df['height'] = df['height'] * 0.01
def data_preprocessing(df):

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
    bmi_mappping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_mappping)

    def categorize_age(age):
        if age < 40:
            return 'Young'
        elif 40 <= age < 60:
            return 'Middle-aged'
        else:
            return 'Senior'

    df['age_group'] = df['age'].apply(categorize_age)
    age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
    df['age_group'] = df['age_group'].map(age_mapping)

    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 'Normal'
        elif ap_hi >= 140 or ap_lo >= 90:
            return 'Hypertension'
        else:
            return 'High-Normal'

    df['bp_category'] = df.apply(lambda row: categorize_bp(
        row['ap_hi'], row['ap_lo']), axis=1)
    bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
    df['bp_category'] = df['bp_category'].map(bp_mapping)

    features = ['bmi', 'bmi_category', 'age_group', 'bp_category', 'cardio']
    numeric_df = df[features]

    return df


processed_df = data_preprocessing(df)

def data_splitting():
    # Title container
    with st.container():
        st.header("Data Splitting")
        st.subheader("This is the preprocessed data set for the model")

        st.write(processed_df.head())
        st.write("Shape:", processed_df.shape)

        # generate 2d classification dataset
        st.subheader("Data splitting and processing")
        st.code("X = df.drop('cardio', axis=1).to_numpy()")
        X = processed_df.drop('cardio', axis=1).to_numpy()

        st.write("X[:2]", X[:2])
        st.write("X.shape:", X.shape)

        st.code("y = df['cardio'].to_numpy()")
        y = processed_df['cardio'].to_numpy()
        st.write("y[:2]", y[:2])
        st.write('y.shape:', y.shape)

        # data Scaling
        st.subheader("Scaling the data...")
        st.code("""scaler = StandardScaler()""")
        scaler = StandardScaler()

        st.code("X = scaler.fit_transform(X)")
        X = scaler.fit_transform(X)
        st.write("X after scaling")
        st.code("X[:2]")
        st.write(X[:2])

        # Data Splitting
        st.subheader("Splitting the data into train and test sets...")
        st.code(
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("- X_train.shape:", X_train.shape)
        st.write("- X_test.shape:", X_test.shape)
        st.write("- y_train.shape:", y_train.shape)
        st.write("- y_test.shape:", y_test.shape)

        # Further Splitting
        st.write(
            "Further splitting the training data into training and validation sets...")
        st.code("x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)")
        x_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        st.write("- x_train.shape:", x_train.shape)
        # st.write("- x_train[:2]", x_train[:2])
        st.write("- x_val.shape:", x_val.shape)
        st.write("- y_train.shape:", y_train.shape)
        st.write("- y_val.shape:", y_val.shape)



def data_preprocessing():
    df = pd.read_csv('../cardio_train.csv', sep=';')
    df.drop('id', axis=1, inplace=True)
    df['age'] = (df['age'] / 365).round().astype('int')
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
    bmi_mappping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_mappping)

    def categorize_age(age):
        if age < 40:
            return 'Young'
        elif 40 <= age < 60:
            return 'Middle-aged'
        else:
            return 'Senior'

    df['age_group'] = df['age'].apply(categorize_age)
    age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
    df['age_group'] = df['age_group'].map(age_mapping)

    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 'Normal'
        elif ap_hi >= 140 or ap_lo >= 90:
            return 'Hypertension'
        else:
            return 'High-Normal'

    df['bp_category'] = df.apply(lambda row: categorize_bp(
        row['ap_hi'], row['ap_lo']), axis=1)
    bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
    df['bp_category'] = df['bp_category'].map(bp_mapping)

    return df


processed_df = data_preprocessing()

# st.title("Model Training")

# st.write("Shape:", processed_df.shape)

X = processed_df.drop('cardio', axis=1).to_numpy()
y = processed_df['cardio'].to_numpy()

# data Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Further Splitting
x_train, x_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)


# Model Building

def model_building():

    def model(input_dim):
        st.header("Neaural Networks Model Building and Training")
        # st.title("Neaural Networks Model Building and Training")
        st.write("Data Splitting and Processing Completed")
        st.code("""model = Sequential([
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
        return model""")
        
        model = Sequential([
            Dense(500, input_dim=input_dim, activation='relu'),
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
    
    input_dim = x_train.shape[1]
    model = model(input_dim)

    # Function to display model summary


    def display_model_summary(model):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        model_summary_text = buffer.getvalue()
        st.text("Model Summary:")
        st.text(model_summary_text)


    # model = model()
    display_model_summary(model)

    # Button to initiate training
    st.code("""def train(model, x_train, y_train, x_val, y_val, X_test, y_test):
        history = model.fit(x_train, y_train, validation_data=(
            x_val, y_val), epochs=20, batch_size=64)
        score = model.evaluate(X_test, y_test, verbose=0)
        return history, score
            """)
    
    def train(model, x_train, y_train, x_val, y_val, X_test, y_test):
        history = model.fit(x_train, y_train, validation_data=(
            x_val, y_val), epochs=20, batch_size=64)
        # score = model.evaluate(X_test, y_test, verbose=0)
        return history
    

    def save(model):
        # Save the model architecture as JSON
        model_json = model.to_json()
        with open("saved_model.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save the model weights
        model.save_weights("saved_model_weights.weights.h5")
        st.write("Model saved")

    if st.button("Train Model"):
        st.write("Training the model...")
        history = train(model, x_train, y_train, x_val, y_val, X_test, y_test)
        # score = model.evaluate(X_test, y_test, verbose=0)
        st.write("Model Training Completed")
        st.text("Evaluating the model on test data...")
        
        # Display evaluation results
        st.subheader("Evaluation Results")

        # Evaluate the model
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Plot training history
        st.subheader("Training History")
        sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], label='train')
        sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot(plt)


        # Evaluate the model on test data
        st.subheader("Model Evaluation on Test Data")
        st.write("Evaluating the model...")
        score = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test loss: {score[0]:.2f}%")
        st.write(f"Test accuracy: {score[1]:.2f}%")
            
        model_instance = model
        save(model_instance)
        


def load_model():
    # Load model architecture from JSON file
    with open('saved_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    
    # Load model weights
    loaded_model.load_weights("saved_model_weights.weights.h5")
    
    return loaded_model
saved_model = load_model()


def model_testing():
    def welcome(): 
        st.title('welcome all')

    welcome()
    
    # defining the function which will make the prediction using  

    def preprocess_data(age, weight, height, cholesterol, gender, gluc, active, smoke, alco, ap_hi, ap_lo):
        test_data = {
            'age': [age],
            'gender': [gender],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        }
        df = pd.DataFrame(test_data)
        
        # Convert specific columns to integers
        int_columns = ['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        df[int_columns] = df[int_columns].astype(int)

        # Data preprocessing
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
        bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        df['bmi_category'] = df['bmi_category'].map(bmi_mapping)

        def categorize_age(age):
            if age < 40:
                return 'Young'
            elif 40 <= age < 60:
                return 'Middle-aged'
            else:
                return 'Senior'

        df['age_group'] = df['age'].apply(categorize_age)
        age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
        df['age_group'] = df['age_group'].map(age_mapping)

        def categorize_bp(ap_hi, ap_lo):
            if ap_hi < 120 and ap_lo < 80:
                return 'Normal'
            elif ap_hi >= 140 or ap_lo >= 90:
                return 'Hypertension'
            else:
                return 'High-Normal'

        df['bp_category'] = df.apply(lambda row: categorize_bp(row['ap_hi'], row['ap_lo']), axis=1)
        bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
        df['bp_category'] = df['bp_category'].map(bp_mapping)

        print(df.columns)
        return df

    def prediction(numeric_df):   
    
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)
        model = saved_model
        y_pred = model.predict(scaled_features)
        class_label = (y_pred >= 0.5).astype(int)
        
        return class_label[0][0]

    # this is the main function in which we define our webpage  
    def main(): 
        # giving the webpage a title 
        st.title("Cardio Vascular Prediction") 
        
        age = st.number_input("Patient's Age in years", min_value=0, max_value=120, value=30)
        weight = st.number_input("Patient's Weight in kgs", min_value=0, max_value=500, value=70)
        height = st.number_input("Patient's Height in cms", min_value=0, max_value=300, value=170)
        gender_options = {"Male": 0, "Female": 1}
        genders = st.selectbox("Patient's Gener", list(gender_options.keys()))
        gender = gender_options[genders]
        gluc = st.selectbox("Patient's Glucose Levels", [0, 1, 2])
        activity_options = {"Not physically active": 0, "Active": 1}
        activity = st.selectbox("Patient's Activity Level", list(activity_options.keys()))
        # Retrieve the numerical value corresponding to the selected option
        active = activity_options[activity]
        
        smoke_options = {"Doesn't smoke": 0, "Active smoker": 1}
        smoking = st.selectbox("Patient's Smoking Habits", list(smoke_options.keys()))
        smoke = smoke_options[smoking]
        cholesterol = st.selectbox("Patient's Cholesterol Levels", [1, 2, 3])
        # alco = st.selectbox("Patient's Alcohol Habits", ["Non-alcoholic", "Alcoholic"])
        alcohol_options = {"Non-alcoholic": 0, "Alcoholic": 1}
        alcohol = st.selectbox("Patient's Alcohol Habits", list(alcohol_options.keys()))
        # Retrieve the numerical value corresponding to the selected option
        alco = alcohol_options[alcohol]
        ap_hi = st.number_input("Patient's Diastolic Blood Pressure", min_value=0, max_value=300, value=120)
        ap_lo = st.number_input("Patient's Systolic Blood Pressure", min_value=0, max_value=300, value=80)
        result = ""
        
        # the below line ensures that when the button called 'Predict' is clicked,  
        
        if st.button("Predict"): 
            numeric_df = preprocess_data(age, weight, height, cholesterol, gender, gluc, active, smoke, alco, ap_hi, ap_lo)
            result = prediction(numeric_df)
            
            # Additional info
            bmi = calculate_bmi(weight, height)
            age_category = categorize_age(int(age))
            
            # Output categorization
            if result == 0:
                prediction_result = "Less likely"
            else:
                prediction_result = "More likely"
            
            # Display result with additional info
            st.success(f"The output is {prediction_result}.\nBMI: {bmi}, Age Category: {age_category}")
        
    main()

 # BMI calculation function
def calculate_bmi(weight, height):
    bmi = float(weight) // ((float(height) / 100) ** 2)
    return bmi

# Age categorization function
def categorize_age(age):
    if age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'
        

# Title container
with st.container():
    st.title("Machine Learning Model to test for cardio Vascular Diseases")

    functions = [analysis_visualizations,data_preprocessing, data_splitting,  model_building, model_testing]
    
    # Initialize session state for the index of the current graph
    if 'current_function_index' not in st.session_state:
        st.session_state.current_function_index = 0
    
    # Display the current graph
    current_function_index = st.session_state.current_function_index
    
    for i in range(len(functions)):
        if i == current_function_index:
            functions[i]()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.current_function_index > 0:
            if st.button('Previous'):
                st.session_state.current_function_index -= 1
                st.experimental_rerun()
    
    with col2:
        if st.session_state.current_function_index < len(functions) -1:
            if st.button('Next'):
                st.session_state.current_function_index += 1
                st.experimental_rerun()
