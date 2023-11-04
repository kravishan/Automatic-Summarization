import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import InputComponent from './InputComponent';
import SummaryComponent from './SummaryComponent';
import RougeScoresTable from './RougeScoresTable'; 

function App() {
  const [customSummary, setCustomSummary] = useState('');
  const [summaries, setSummaries] = useState({});
  const [rougeScores, setRougeScores] = useState({});
  const [goldenSummary, setGoldenSummary] = useState('');
  const [customRougeScores, setCustomRougeScores] = useState({});
  const [dataFetched, setDataFetched] = useState(false);


  const handleSummarize = (text) => {
    // Send the text to OpenAI for summarization
    const apiKey = process.env.REACT_APP_OPENAI_API_KEY; 
    console.log('Enter text=',text);

        // Send the ChatGPT text to local API 
        axios
      .post(
        'https://api.openai.com/v1/engines/davinci/completions',
        {
          prompt: `Only provide the summary of the following text: ${text}`,
          max_tokens: 50,
        },
        {
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
        },
      )
      .then((openAIResponse) => {
        const summarizedText = openAIResponse.data.choices[0].text.trim();
        axios
          .post('http://127.0.0.1:5000/summarize', { golden_summary: summarizedText, input_text: text })
          .then((localApiResponse) => {
            const result = localApiResponse.data;
            setCustomSummary(result.custom_summary);
            setSummaries(result.summaries);
            setRougeScores(result.rouge_scores);
            setGoldenSummary(result.golden_summary);
            setCustomRougeScores(result.custom_rouge_scores);
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
          {dataFetched && (
            <div>
              <h2>Custom Summary</h2>
              <div className="summary-box">
                <SummaryComponent summary={customSummary} />
                <h3>Custom Rouge Scores</h3>
                {/* Here, we can display the custom Rouge scores */}
                <div className="summary-box">
                  <RougeScoresTable rougeScores={customRougeScores} />
                </div>
              </div>
              <h2>Golden Summary</h2>
              <div className="summary-box">
                <SummaryComponent summary={goldenSummary} />
              </div>
              {Object.keys(summaries).map((approach) => (
                <div key={approach}>
                  <h3>{approach} Summary</h3>
                  <div className="summary-box">
                    <SummaryComponent summary={summaries[approach]} />
                    <h3>{approach} Rouge Scores</h3>
                    {/* Display Rouge scores for each summarization approach */}
                    <div className="summary-box">
                      <RougeScoresTable rougeScores={rougeScores[approach]} />
                    </div>
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
