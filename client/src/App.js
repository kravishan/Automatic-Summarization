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
    // For testing purposes, I have used a static text
    text = "Sri Lanka, officially known as the Democratic Socialist Republic of Sri Lanka, is a tropical paradise located in the Indian Ocean, just off the southern coast of India. This captivating island nation, often referred to as the Teardrop of India, is a place of incredible natural beauty, a rich and complex history, diverse cultures, and a warm and welcoming population. Covering an area of approximately 65,610 square kilometers, Sri Lanka is home to more than 21 million people, making it one of the most densely populated countries in the world. Sri Lanka's geographical diversity is a striking feature. The island is characterized by a varied topography that ranges from the pristine beaches of the coastline to the central highlands, where the terrain climbs to over 2,500 meters above sea level. The central highlands are adorned with tea plantations, lush forests, and cool, misty air, making it a stark contrast to the warm and tropical climate of the coastal regions. The monsoon seasons influence the weather, with the southwest monsoon bringing rain to the western and southern coasts, while the northeast monsoon affects the east and north. Sri Lanka's geography is as diverse as it is breathtaking. Sri Lanka has a history that spans more than 2,500 years. Its roots can be traced back to the ancient Sinhalese kingdom of Anuradhapura, which was one of the world's earliest centers of Buddhism. The island has also been influenced by Indian, Southeast Asian, and European cultures throughout its history. Remarkable historical sites, such as the ancient cities of Anuradhapura and Polonnaruwa, the rock fortress of Sigiriya, and the sacred city of Kandy, tell the tale of Sri Lanka's rich past. The island's cultural tapestry is a vibrant and harmonious blend of various ethnicities and religions. While the majority of the population practices Buddhism, there are significant Hindu, Muslim, and Christian communities. Festivals like Vesak, the Sinhala and Tamil New Year, and Diwali are celebrated with great enthusiasm, showcasing the diverse cultural traditions. Sri Lanka is a biodiversity hotspot, known for its incredible flora and fauna. The island is home to a multitude of ecosystems, from the lush rainforests of Sinharaja to the arid plains of the north. Its wildlife is equally diverse, with iconic species such as the Sri Lankan leopard, Asian elephant, sloth bear, and a myriad of bird species, many of which are endemic. National parks like Yala, Wilpattu, and Udawalawe offer visitors a chance to observe these creatures in their natural habitat. The waters surrounding Sri Lanka are also a hotspot for marine life, attracting enthusiasts for whale and dolphin watching. In recent years, Sri Lanka has seen a surge in tourism, as travelers from around the world discover the island's beauty and charm. The country has invested heavily in its tourism infrastructure, with a wide range of accommodations to cater to all budgets and tastes. From luxury resorts and boutique hotels to eco-friendly lodges and budget guesthouses, there is something for every traveler. Sri Lanka's historical and cultural attractions draw visitors to explore the ancient cities, sacred temples, and colonial architecture. The ancient city of Polonnaruwa, the Temple of the Sacred Tooth Relic in Kandy, and the colonial-era fort in Galle are some of the top destinations. The island's pristine beaches, such as Mirissa, Unawatuna, and Arugam Bay, are perfect for relaxation and water sports. For the adventurous, hiking through the central highlands, particularly to World's End in Horton Plains National Park, offers breathtaking views. Sri Lanka's culinary scene is an adventure in itself. The country is known for its spicy curries, fresh seafood, and a wide range of tropical fruits. Trying local dishes like rice and curry, hoppers, and kottu roti is a delightful experience for food lovers. The island is also famous for its tea, with the central highlands producing some of the world's finest Ceylon tea. Sri Lanka continues to evolve, seeking a bright future while preserving its cultural heritage and natural wonders. The nation has faced challenges, including a prolonged civil conflict that ended in 2009 and the devastating impact of the 2004 Indian Ocean tsunami. However, it has shown resilience and determination to rebuild and grow. The government's commitment to sustainable tourism and conservation efforts, such as reforestation and wildlife protection, indicate a dedication to preserving Sri Lanka's natural beauty for future generations. In conclusion, Sri Lanka is a multifaceted jewel that offers a unique blend of history, culture, and natural beauty. It's a place where ancient traditions coexist with modernity, where diverse landscapes await exploration, and where the warmth and hospitality of its people leave an indelible mark on every visitor. As you embark on a journey through this resplendent island, you will discover a world of wonders, from the lush forests to the pristine shores, from ancient temples to vibrant markets, and from the captivating wildlife to the flavorful cuisine. Sri Lanka is a destination that promises to capture your heart and soul.";

    console.log('Enter text=', text);
  
    axios
      .post('https://api.openai.com/v1/engines/davinci/completions', {
        prompt: `Summarize: ${text}`,
        max_tokens: 500,
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