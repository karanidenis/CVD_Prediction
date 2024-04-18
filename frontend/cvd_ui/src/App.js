import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [age, setAge] = useState();
  const [height, setHeight] = useState(0);
  const [weight, setWeight] = useState();
  const [gender, setGender] = useState('Male');
  const [alco, setAlcohol] = useState('Non-alcoholic');
  const [smoke, setSmoking] = useState("Doesn't smoke");
  const [ap_hi, setApHi] = useState(0);
  const [ap_lo, setApLo] = useState(0);
  const [active, setActivity] = useState('Not physically active');
  const [gluc, setGluc] = useState(0);
  const [cholesterol, setCholesterol] = useState('Normal');
  const [result, setResult] = useState('');
  const [isDataReady, setIsDataReady] = useState(false);
  const [confirmationMessage, setConfirmationMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);



  useEffect(() => {
    // Check if all fields are filled out
    if (
      age !== '' &&
      weight !== '' &&
      height !== '' &&
      gender !== '' &&
      gluc !== '' &&
      active !== '' &&
      smoke !== '' &&
      cholesterol !== '' &&
      alco !== '' &&
      ap_hi !== '' &&
      ap_lo !== ''
    ) {
      setIsDataReady(true);
    } else {
      setIsDataReady(false);
    }
  }, [age, height, weight, gender,alco, smoke, ap_hi, ap_lo, active, gluc, cholesterol,]);

  
  const preprocessData = (data) => {
    // Convert string values to appropriate types
    const age = parseInt(data.age);
    const height = parseInt(data.height);
    const weight = parseInt(data.weight);
    const gender = data.gender;
    const alco = data.alco === 'Alcoholic' ? 1 : 0; // Convert to integer directly
    const smoke = data.smoke === 'Active smoker' ? 1 : 0;
    const ap_hi = parseInt(data.ap_hi);
    const ap_lo = parseInt(data.ap_lo);
    const active = data.active === 'Active' ? 1 : 0;
    const gluc = parseInt(data.gluc);
    const cholesterol = data.cholesterol === 'Normal' ? 1 : data.cholesterol === 'Above normal' ? 2 : 3;
  
    // Return the preprocessed data in the desired format
    return {
      age,
      height,
      weight,
      gender,
      alco,
      smoke,
      ap_hi,
      ap_lo,
      active,
      gluc,
      cholesterol,
    };
  };
  
  const handleConfirmInput = async () => {
    try {
      // Preprocess data
      const processedData = preprocessData({
        age,
        height,
        weight,
        gender,
        alco,
        smoke,
        ap_hi,
        ap_lo,
        active,
        gluc,
        cholesterol,
      });

      console.log('Data being sent:', processedData);

      const response = await fetch('http://127.0.0.1:8080/confirm-input', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(processedData)
      });

      if (response.ok) {
        // Data successfully sent and ready for prediction
        const message = 'Data successfully sent and ready for prediction';
        console.log(message);
        setIsDataReady(true);
        setConfirmationMessage(message);
      } else {
        // Display error message
        const errorMessage = `Error: ${response.statusText}`;
        console.error(errorMessage);
        setIsDataReady(false);
        setConfirmationMessage(errorMessage);
      }
    } catch (error) {
      // Display error message
      const errorMessage = `Error: ${error.message}`;
      console.error(errorMessage);
      setIsDataReady(false);
      setConfirmationMessage(errorMessage);
    }
};
  
// const handleSubmit = async (e) => {
//   e.preventDefault();
 

const handleSubmit = async (e) => {
  e.preventDefault();

  try {
    // Call handleConfirmInput to preprocess data
    await handleConfirmInput();

    // Use preprocessed data
    const data = preprocessData({
      age,
      height,
      weight,
      gender,
      alco,
      smoke,
      ap_hi,
      ap_lo,
      active,
      gluc,
      cholesterol,
    });

    // Log data before sending the request
    console.log('Data being sent:', data);

    setIsLoading(true); // Set loading state while waiting for response

    const response = await fetch('http://127.0.0.1:8080/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    setIsLoading(false); // Reset loading state after response received

    if (response.ok) {
      // Check if the response body has been consumed
      if (!response.bodyUsed) {
        const responseBody = await response.text();
        const prediction = JSON.parse(responseBody).prediction;
        setResult(prediction);
        console.log(prediction);
      } else {
        console.error('Error: Response body already read.');
        setResult('');
      }
    } else {
      // Display error message
      console.error('Error:', response.statusText);
      setResult('');
    }
  } catch (error) {
    // Display error message
    console.error('Error:', error);
    setResult('');
    setIsLoading(false); // Reset loading state in case of error
  }
};



  return (
    <div className="App">
      <h1>Cardio Vascular Prediction</h1>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <label>
            Patient's Gender:
            <select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </label>

          <label>
            Patient's Age in years:
            <input type="number" value={age} onChange={(e) => setAge(e.target.value)} min="0" max="120" />
          </label>
        </div>

        <div className="input-group">
          <label>
            Patient's Weight in kgs:
            <input type="number" value={weight} onChange={(e) => setWeight(e.target.value)} min="0" max="500" />
          </label>

          <label>
            Patient's Height in cms:
            <input type="number" value={height} onChange={(e) => setHeight(e.target.value)} min="0" max="300" />
          </label>
        </div>

        <div className="input-group">
          <label>
            Patient's Diastolic blood pressure:
            <input type="number" value={ap_lo} onChange={(e) => setApLo(e.target.value)} min="0" max="300" />
          </label>

          <label>
            Patient's Systolic blood pressure:
            <input type="number" value={ap_hi} onChange={(e) => setApHi(e.target.value)} min="0" max="300" />
          </label>
        </div>

        <div className="input-group">
          <label>
            Patient's Alcoholic habits:
            <select value={alco} onChange={(e) => setAlcohol(e.target.value)}>
              <option value="Alcoholic">Alcoholic</option>
              <option value="Non-alcoholic">Non-alcoholic</option>
            </select>
          </label>

          <label>
            Patient's cholesterol levels:
            <select value={cholesterol} onChange={(e) => setCholesterol(e.target.value)}>
              <option value="Normal">Normal</option>
              <option value="Above normal">Above normal</option>
              <option value="Well above normal">Well above normal</option>
            </select>
          </label>
        </div>

        <div className="input-group">
          <label>
            Patient's Glucose Levels:
            <select value={gluc} onChange={(e) => setGluc(e.target.value)}>
              <option value="Normal">Normal</option>
              <option value="Above normal">Above normal</option>
              <option value="Well above normal">Well above normal</option>
            </select>
          </label>

          <label>
            Patient's Smoking habits:
            <select value={smoke} onChange={(e) => setSmoking(e.target.value)}>
              <option value="Active smoker">Active</option>
              <option value="Non-smoker">Non-smoker</option>
            </select>
          </label>
        </div>

        <div className="input-group">
          <label>
            Patient's Physical Activity:
            <select value={active} onChange={(e) => setActivity(e.target.value)}>
              <option value="Active">Active</option>
              <option value="Not active">Not active</option>
            </select>
          </label>
        </div>
      
        <div className="button-group">
          <button type="button" onClick={handleConfirmInput}>Submit</button>
          {confirmationMessage && <div>{confirmationMessage}</div>}
          {isLoading ? (
            <div>Loading...</div>
          ) : (
            <button type="submit">Predict</button>
          )}
        </div>
      </form>
      {result && <div>Prediction Result: {result}</div>}
    </div>
  );
}

export default App;
