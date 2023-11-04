import React from 'react';

const RougeScoresTable = ({ rougeScores }) => {
  return (
    <table className="table-bordered">
      <thead>
        <tr>
          <th>Measure</th>
          <th>F-score</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(rougeScores).map(([key, { f, p, r }]) => (
          <tr key={key}>
            <td>{key}</td>
            <td>{f.toFixed(4)}</td>
            <td>{p.toFixed(4)}</td>
            <td>{r.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default RougeScoresTable;
