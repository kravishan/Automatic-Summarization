import React from 'react';

function SummaryComponent({ summary }) {
  return (
    <div className="fixed-box">
      <h4>Summary:</h4>
      <p>{summary}</p>
    </div>
  );
}

export default SummaryComponent;