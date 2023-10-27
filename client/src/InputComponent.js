// InputComponent.js
import React, { useState } from 'react';

function InputComponent(props) {
  const [text, setText] = useState('');

  const handleSummarize = () => {
    props.onSummarize(text);
  };

  return (
    <div className="InputComponent">
      <textarea
        placeholder="Enter text to summarize..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      ></textarea>
      <button className="SummarizeButton" onClick={handleSummarize}>
        Summarize
      </button>
    </div>
  );
}

export default InputComponent;
