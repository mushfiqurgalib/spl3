import React, { useState } from "react";
import axios from "axios";
import image1 from "../../src/images/ans1.jpg";
import { ThreeDots } from "react-loader-spinner";
import "./style.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const submitHandler = (e) => {
    e.preventDefault();
    if (selectedFile) {
      setLoading(true);

      const formData = new FormData();
      formData.append("file", selectedFile);

      axios
        .post("http://127.0.0.1:5000/upload", formData, {
          responseType: "arraybuffer", // Ensure binary response
        })
        .then((response) => {
          if (response.status === 200) {
            console.log("Image uploaded successfully");

            // Create a blob from the binary response data
            const blob = new Blob([response.data], { type: "image/jpeg" });

            // Convert the blob to an image URL
            const imageUrl = URL.createObjectURL(blob);

            // Set the processed image URL to display it
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
    <div className="App">
      <h1>Image Upload and Processing</h1>
      <form
        onSubmit={submitHandler}
        encType="multipart/form-data"
        id="imageForm"
      >
        <input type="file" onChange={handleFileChange} />
        <button type="submit" className="btn btn-success ">
          Upload
        </button>
      </form>

      {loading && (
         <div className="loader-container">
        <ThreeDots
          display= "flex"
          justify-content= "center"
          align-items= "center"
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
          <h2>Uploaded Image:</h2>
          <img src={image1} alt="Uploaded" />
        </div>
      )}

      {processedImage && (
        <div>
          <h2>Processed Image:</h2>
          <img src={processedImage} alt="Processed" />
        </div>
      )}
    </div>
  );
}

export default App;
