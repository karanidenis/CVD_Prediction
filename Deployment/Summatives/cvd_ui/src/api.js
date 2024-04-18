// Function to make a POST request to the backend server
async function makePredictionRequest(inputData) {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(inputData)
    });
    const data = await response.json();
    return data.prediction;
  }
  
  // Function to handle user input and make predictions
  async function handlePrediction() {
    // Get user input data
    const age = document.getElementById('age').value;
    const weight = document.getElementById('weight').value;
    // Get other input values similarly
    
    // Make prediction request to the backend server
    const prediction = await makePredictionRequest({ age, weight, /* other input values */ });
  
    // Display prediction result
    const resultElement = document.getElementById('predictionResult');
    resultElement.innerText = `Prediction: ${prediction}`;
  }
  