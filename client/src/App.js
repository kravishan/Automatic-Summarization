import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import InputComponent from './InputComponent';
import SummaryComponent from './SummaryComponent';

function App() {
  const [customSummary, setCustomSummary] = useState('');
  const [chatGPTSummary, setChatGPTSummary] = useState('');
  const [sumySummaries, setSumySummaries] = useState({});
  const [modifiedText, setModifiedText] = useState('');
  const [dataFetched, setDataFetched] = useState(false);

  const apiKey = process.env.REACT_APP_OPENAI_API_KEY;

  const handleSummarize = (text) => {
    // Send the text to OpenAI for summarization
    console.log('Enter text=', text);
    console.log(process.env);

    axios
      .post('https://api.openai.com/v1/engines/davinci/completions', {
        prompt: `Summarize: ${text}`,
        max_tokens: 50,
      }, {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        }
      })
      .then((openAIResponse) => {
        const summarizedText = openAIResponse.data.choices[0].text;
        console.log('ChatGPT generated text=', summarizedText);
        setChatGPTSummary(summarizedText);

        // Send the ChatGPT text to local API
        axios
          .post('http://127.0.0.1:5000/summarize', { text: summarizedText })
          .then((localApiResponse) => {
            const result = localApiResponse.data;
            setCustomSummary(result.custom_summary);
            setSumySummaries(result.sumy_summaries);
            setModifiedText(result.modified_text);
            setDataFetched(true);
          })
          .catch((error) => {
            console.error('Error in local API request:', error);
          });
      })
      .catch((error) => {
        console.error('Error in OpenAI request:', error);
      });
  };

  return (
    <div className="scroll-container">
      <div className="App">
        <h1>Text Summarization Tool</h1>
        <InputComponent onSummarize={handleSummarize} />
        <div className="Result">
        {chatGPTSummary && (
            <div>
              <h2>Golden Summary</h2>
              <div className="summary-box">
                <SummaryComponent summary={chatGPTSummary} />
              </div>
            </div>
          )}
          {dataFetched && (
            <div>
              <h2>Custom Summary</h2>
              <div className="summary-box">
                <SummaryComponent summary={customSummary} />
              </div>
            </div>
          )}         
          {dataFetched && (
            <div>
              <h2>Sumy-Based Summaries</h2>
              {Object.keys(sumySummaries).map((approach) => (
                <div key={approach}>
                  <h3>{approach}</h3>
                  <div className="summary-box">
                    <SummaryComponent summary={sumySummaries[approach].join(' ')} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
