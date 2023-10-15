import React, { useState } from 'react';

function SummaryComponent({ summary }) {
  const [isExpanded, setExpanded] = useState(false);

  const toggleExpand = () => {
    setExpanded(!isExpanded);
  };

  return (
    <div className="summary-box">
      <div
        className="summary-content"
        style={{
          maxHeight: isExpanded ? 'none' : '50px',
          overflow: 'hidden',
          transition: 'max-height 0.5s',
        }}
      >
        <p>{summary}</p>
      </div>
      {!isExpanded && (
        <button onClick={toggleExpand} className="more-button">
          More
        </button>
      )}
    </div>
  );
}

export default SummaryComponent;
