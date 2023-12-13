import React, { useState, useEffect } from 'react';
import axios from 'axios';

const History = () => {
  const [imageList, setImageList] = useState([]);
  const username = localStorage.getItem('username');

  useEffect(() => {
    if (!username) {
      console.error('Username not found in localStorage');
      return;
    }

    // Make a request to the backend to get the list of images
    axios.get(`http://127.0.0.1:5000/get_images?username=${username}`)
      .then(response => {
        const imageListData = response.data;
        setImageList(imageListData);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, [username]);

  return (
    <div>
      <h2>Images for Username: {username}</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Percentage</th>
            <th>Current Time</th>
            <th>Image</th>
          </tr>
        </thead>
        <tbody>
          {imageList.map(image => (
            <tr key={image._id}>
              <td>{image._id}</td>
              <td>{image.percentage}</td>
              <td>{new Date(image.current_time).toLocaleString()}</td>
              <td><img src={image.url} alt="Uploaded" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default History;
