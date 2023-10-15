import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import InputComponent from './InputComponent';
import SummaryComponent from './SummaryComponent';

function App() {
  const [summary, setSummary] = useState('');
  const [modifiedText, setModifiedText] = useState(''); // To store text improved by backend

  const handleSummarize = (text) => {
    console.log('Button clicked, sending request');
    axios
      .post('http://127.0.0.1:5000/summarize', { text })
      .then((response) => {
        const result = response.data;
        setSummary(result.summary);
        setModifiedText(result.modified_text);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="App">
      <h1>Text Summarization Tool</h1>
      <InputComponent onSummarize={handleSummarize} />
      <div className="Result">
        <SummaryComponent summary={modifiedText} />
      </div>
    </div>
  );
}

export default App;
