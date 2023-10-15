import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import InputComponent from './InputComponent';
import SummaryComponent from './SummaryComponent';

function App() {
  const [customSummary, setCustomSummary] = useState('');
  const [sumySummaries, setSumySummaries] = useState({});
  const [modifiedText, setModifiedText] = useState(''); // To store text improved by backend

  const handleSummarize = (text) => {
    console.log('Button clicked, sending request');
    axios
      .post('http://127.0.0.1:5000/summarize', { text })
      .then((response) => {
        const result = response.data;
        setCustomSummary(result.custom_summary);
        setSumySummaries(result.sumy_summaries);
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
        <div>
          <h2>Custom Summary</h2>
          <SummaryComponent summary={customSummary} />
        </div>
        <div>
          <h2>Sumy-Based Summaries</h2>
          {Object.keys(sumySummaries).map((approach) => (
            <div key={approach}>
              <h3>{approach}</h3>
              <SummaryComponent summary={sumySummaries[approach].join(' ')} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
