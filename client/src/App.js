import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import InputComponent from './InputComponent';
import SummaryComponent from './SummaryComponent';
import RougeScoresComponent from './RougeScoresComponent';

function App() {
  const [customSummary, setCustomSummary] = useState('');
  const [chatGPTSummary, setChatGPTSummary] = useState('');
  const [sumySummaries, setSumySummaries] = useState({});
  const [dataFetched, setDataFetched] = useState(false);
  const [rougeScores, setRougeScores] = useState([]);

  const apiKey = process.env.REACT_APP_OPENAI_API_KEY;

  const handleSummarize = (text) => {
    // Send the text to OpenAI for summarization
    console.log('Enter text=', text);
  
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
        setChatGPTSummary(summarizedText);
        console.log('ChatGPT generated text=', summarizedText);

        // Send the ChatGPT text to the local API
        axios
          .post('http://127.0.0.1:5000/summarize', { text: summarizedText })
          .then((localApiResponse) => {
            const result = localApiResponse.data;
            setCustomSummary(result.custom_summary);
            setSumySummaries(result.sumy_summaries);
            setRougeScores(result.rouge_scores);

            // Log the values to the console
            console.log('Custom Summary:', result.custom_summary);
            console.log('Sumy Summaries:', result.sumy_summaries);
            console.log('Rouge Scores:', result.rouge_scores);

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
    {sumySummaries &&
      Object.keys(sumySummaries).map((approach) => {
        //console.log('Approach:', approach);

        const summaries = sumySummaries[approach];
        //console.log('Summaries:', summaries); 

        const scores = rougeScores?.find((scores) => scores.approach === approach);
        console.log('rougeScores:', rougeScores);


        //console.log('Rouge Scores:', scores); 

        return (
          <div key={approach}>
            <h3>{approach}</h3>
            <div className="summary-box">
              {Array.isArray(summaries) && summaries.length > 0 ? (
                <SummaryComponent summary={summaries.join(' ')} />
              ) : (
                <p>No summary available for this approach</p>
              )}
            </div>
            {scores ? (
              <div>
                <RougeScoresComponent rougeScores={scores.scores} />
              </div>
            ) : (
              <p>Rouge scores not available for this approach</p>
            )}
          </div>
        );
      })}
  </div>
)}

        </div>
      </div>
    </div>
  );
}

export default App;