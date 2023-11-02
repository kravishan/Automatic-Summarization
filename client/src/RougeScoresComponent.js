import React from 'react';

function RougeScoresComponent({ rougeScores }) {
  return (
    <div>
      {rougeScores.map(scores => (
        <div key={scores.approach}>
          <h4>{scores.approach}</h4>
          <p>ROUGE-1: {scores['rouge-1']}</p>
          <p>ROUGE-2: {scores['rouge-2']}</p>
          <p>ROUGE-L: {scores['rouge-l']}</p>
        </div>
      ))}
    </div>
  );
}

export default RougeScoresComponent;
