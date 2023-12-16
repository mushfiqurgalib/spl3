import React, { useState } from "react";
import axios from "axios";
import image1 from "../../src/images/ans1.jpg";
import { ThreeDots } from "react-loader-spinner";
import { useNavigate } from 'react-router-dom';

function App() {
  const navigate = useNavigate();
  const [fileName, setFileName] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tumorpercentage,settumorpercentage] = useState("");
  
  const handleLogout = () => {
    localStorage.removeItem('username');
    navigate('/login')
  }

  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
  
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
    } else {
      // Handle the case where no file is selected or the selection is canceled
      setSelectedFile(null);
      setFileName("");
    }
  };
  

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
  };
  const handleUploadButtonClick = () => {
    uploadFile();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setSelectedFile(droppedFile);
    setFileName(droppedFile.name);
  };

  const submitHandler = (e) => {
    e.preventDefault();
    uploadFile();
  };

  const uploadFile = () => {
    if (selectedFile) {
      setLoading(true);
  
      // Retrieve the username from localStorage
      const username = localStorage.getItem('username');
  
      const formData = new FormData();
      formData.append("file", selectedFile);
  
      // Include the username in the request headers
      const config = {
        headers: {
          Authorization: username,
          'Content-Type': 'multipart/form-data',
        },
      };
  
      axios
        .post("http://127.0.0.1:5000/upload", formData, config)
        .then((response) => {
          if (response.status === 200) {
            console.log("Image uploaded successfully");
            console.log(response.data);
            const imageUrl = `data:image/jpeg;base64,${response.data.image}`;
            const percentage = response.data.percentage;
            settumorpercentage(percentage);
            setProcessedImage(imageUrl);
          }
        })
        .catch((error) => {
          console.error(error);
          if (error.response) {
            console.log(error.response);
            if (error.response.status === 401) {
              alert("Invalid credentials");
            }
          }
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      console.log("No file selected");
    }
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
          <a className="nav-link active" aria-current="page" href="/"  style={{ color: 'white' }}>Home</a>
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
    <div className="App">
      <h1>Brain MRI Segmentation</h1>
      <form
        onSubmit={submitHandler}
        encType="multipart/form-data"
        id="imageForm"
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-area-container" >
          <div className="drop-area">
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: "none" }}
          />
    <p style={{ display: "inline-block", whiteSpace: "nowrap" }}>
  {fileName
    ? `Selected File: ${fileName}`
    : "Drag & Drop your image here or click to select"}
</p>


          </div>
        </div>
        {/* <button type="submit" className="btn btn-success">
          Upload
        </button> */}
      </form>

      <div>
  {/* Button for selecting files */}
  <label htmlFor="fileInput" className="file-input-label">
    <button
      className="btn btn-primary file-input-button"
      onClick={() => document.getElementById("fileInput").click()}
    >
      Select File
    </button>
  </label>
  <input
    type="file"
    id="fileInput"
    onChange={handleFileChange}
    accept="image/*"
    style={{ display: "none" }}
  />

  <button onClick={handleUploadButtonClick} className="btn btn-success">
    Segment
  </button>
</div>



      {loading && (
        <div className="loader-container">
          <ThreeDots
            display="flex"
            justifyContent="center"
            alignItems="center"
            height="80"
            width="80"
            radius="9"
            color="#4fa94d"
            ariaLabel="three-dots-loading"
            wrapperStyle={{}}
            wrapperClassName=""
            visible={true}
          />
        </div>
      )}

{processedImage && (
  <div>
    <div>
      {tumorpercentage != null ? (
        <div>
          <p>
            <b>Tumor percentage: </b>
            <span style={{ color: tumorpercentage < 5 ? 'green' : 'red' }}>
              {tumorpercentage.toFixed(2)}
            </span>
            {/* <span style={{ color: tumorpercentage < 5 ? 'green' : 'red' }}>
              {tumorpercentage < 5 ? ' (Not Severe)' : ' (Severe)'}
            </span> */}
          </p>
        </div>
      ) : (
        ""
      )}
    </div>

    <table>
      <tbody>
        <tr>
          <th>MRI Image:</th>
          <th>Predicted Brain Tumor:</th>
        </tr>
        <tr>
          <td><img src={image1} alt="Uploaded" /></td>
          <td><img src={processedImage} alt="Processed" /></td>
        </tr>
        <tr>
       

        </tr>
      </tbody>
    </table>
  </div>
)}
</div>
</>
)}

export default App;
