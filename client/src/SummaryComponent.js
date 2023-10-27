import React from 'react';

function SummaryComponent({ summary }) {
  return (
    <div className="fixed-box">
      <h2>Summary:</h2>
      <p>{summary}</p>
    </div>
  );
}

export default SummaryComponent;