import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto'
import { saveAs } from 'file-saver';

const History = () => {
  const navigate = useNavigate();
  const [imageList, setImageList] = useState([]);
  const username = localStorage.getItem('username');
  const handleLogout = () => {
    localStorage.removeItem('username');
    navigate('/');
  };

  useEffect(() => {
    if (!username) {
      console.error('Username not found in localStorage');
      return;
    }

    // Make a request to the backend to get the list of images
    axios
      .get(`http://127.0.0.1:5000/get_images?username=${username}`)
      .then((response) => {
        const imageListData = response.data;
        setImageList(imageListData);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  }, [username]);

  const chartData = {
    labels: imageList.map((image) => new Date(image.current_time).toLocaleString()),
    datasets: [
      {
        label: 'Tumor Percentage',
        data: imageList.map((image) => image.percentage),
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    
    scales: {
      x: {
        type: 'category',
        position: 'bottom',
      },
      y: {
        beginAtZero: true,
        max: 20,
      },
    },
  };
  const exportChart = () => {
    // Access the chart canvas
    const chartCanvas = document.getElementById('myChart');

    // Convert the chart canvas to a data URL
    const dataUrl = chartCanvas.toDataURL('image/png');

    // Save the data URL as a file using file-saver
    saveAs(dataUrl, 'chart.png');
  };
  return (
    <>
     <nav className="navbar navbar-expand-lg bg-primary">
  <div className="container-fluid">
    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span className="navbar-toggler-icon"></span>
    </button>
    <div className="collapse navbar-collapse" id="navbarSupportedContent">
      <ul className="navbar-nav me-auto mb-2 mb-lg-0">
        <li className="nav-item">
          <a className="nav-link active" aria-current="page" href="/image"  style={{ color: 'white' }}>Home</a>
        </li>
        <li className="nav-item">
          <a className="nav-link" href="/history"  style={{ color: 'white' }}>History</a>
        </li>
        
      </ul>
      <ul className="navbar-nav ml-auto">
        <li className="nav-item">
          <button className="btn-danger" onClick={handleLogout} >Logout</button>
        </li>
      </ul>
    </div>
  </div>
</nav>
<div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '50%', width: '50%', margin: '0 auto' }}>
        <h2>Images for Username: {username}</h2>
        <Bar id="myChart"data={chartData} options={options} />
        <button onClick={exportChart}>Export Chart</button>
        <table>
          <thead>
            <tr>
            <th>Upload Time</th>
              <th>Tumor Percentage</th>
              
              <th>Segmented Image</th>
            </tr>
          </thead>
          <tbody>
            {imageList.map((image) => (
              <tr key={image._id}>
                   <td>{new Date(image.current_time).toLocaleString()}</td>
                <td>{image.percentage.toFixed(2)}</td>
             
                <td>
                  <img src={image.url} alt="Uploaded" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
};

export default History;
