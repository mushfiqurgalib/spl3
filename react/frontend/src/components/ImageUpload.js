import React, { useState } from "react";
import axios from "axios";
import image1 from "../../src/images/ans1.jpg";
import { ThreeDots } from "react-loader-spinner";
import "./style.css";

function App() {
  const [fileName, setFileName] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showDragDropFileName, setShowDragDropFileName] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setFileName(file.name);
    setShowDragDropFileName(false);
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

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setSelectedFile(droppedFile);
    setFileName(droppedFile.name);
    setShowDragDropFileName(true);
  };

  const handleUploadButtonClick = () => {
    uploadFile();
  };

  const uploadFile = () => {
    // Upload logic
  };

  return (
    <div className="App">
      <h1>Brain MRI Segmentation</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          uploadFile();
        }}
        encType="multipart/form-data"
        id="imageForm"
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-area">
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: "none" }}
          />
          <p>
            {!showDragDropFileName && fileName
              ? `Selected File: ${fileName}`
              : "Drag & Drop your image here or click to select"}
          </p>
        </div>
        <button onClick={handleUploadButtonClick} className="btn btn-success">
          Upload
        </button>
      </form>

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
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
