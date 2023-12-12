import React from 'react';
import { Bar } from 'react-chartjs-2';

const MetricsChart = ({ precision, f1Score }) => {
  const data = {
    labels: ['Precision', 'F1 Score'],
    datasets: [
      {
        label: 'Metrics',
        backgroundColor: ['rgba(75,192,192,0.4)', 'rgba(255,99,132,0.4)'],
        borderColor: ['rgba(75,192,192,1)', 'rgba(255,99,132,1)'],
        borderWidth: 1,
        hoverBackgroundColor: ['rgba(75,192,192,0.6)', 'rgba(255,99,132,0.6)'],
        hoverBorderColor: ['rgba(75,192,192,1)', 'rgba(255,99,132,1)'],
        data: [precision, f1Score], // Pass precision and f1Score as props
      },
    ],
  };

  const options = {
    scales: {
      y: {
        type: 'linear', // Specify the type of scale explicitly
        beginAtZero: true,
        max: 100, // Assuming percentage values
      },
    },
  };

  return (
    <div>
      <h2>Precision and F1 Score</h2>
      <Bar data={data} options={options} />
    </div>
  );
};

export default MetricsChart;
