import React, { useState } from "react";
import axios from "axios";

function ImageUpload() {
  const [state, setState] = useState({
    image: null,
    responseMsg: {
      status: "",
      message: "",
      error: "",
    },
  });

  const handleChange = (e) => {
    const imagesArray = [];

    for (let i = 0; i < e.target.files.length; i++) {
      imagesArray.push(e.target.files[i]);
    }

    setState((prevState) => ({
      ...prevState,
      image: imagesArray,
    }));
  };

  const submitHandler = (e) => {
    e.preventDefault();
    const data = new FormData();

    for (let i = 0; i < state.image.length; i++) {
      data.append("file", state.image[i]);
    }

    axios
      .post("http://127.0.0.1:5000/upload", data)
      .then((response) => {
        if (response.status === 201) {
          setState((prevState) => ({
            ...prevState,
            responseMsg: {
              status: response.data.status,
              message: response.data.message,
            },
          }));
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
      });
  };

  return (
    <div className="container py-5">
      <div className="row">
        <div className="col-lg-12">
          <form onSubmit={submitHandler} encType="multipart/form-data" id="imageForm">
            <div className="card shadow">
              {state.responseMsg.status === "successs" ? (
                <div className="alert alert-success">
                  {state.responseMsg.message}
                </div>
              ) : state.responseMsg.status === "failed" ? (
                <div className="alert alert-danger">
                  {state.responseMsg.message}
                </div>
              ) : (
                ""
              )}
              <div className="card-header">
                <h4 className="card-title fw-bold">
                  React-JS and Python Flask Multiple Image Upload with Show Uploaded Images
                </h4>
              </div>
              <div className="card-body">
                <div className="form-group py-2">
                  <label htmlFor="images">Images</label>
                  <input
                    type="file"
                    name="image"
                    multiple
                    onChange={handleChange}
                    className="form-control"
                  />
                  <span className="text-danger">{state.responseMsg.error}</span>
                </div>
                {state.image && state.image.length > 0 && (
                  <div className="uploaded-image">
                    <h5>Last Uploaded Image:</h5>
                    <img
                      src={URL.createObjectURL(state.image[state.image.length - 1])}
                      alt="Last Uploaded"
                      style={{ maxWidth: "100%", maxHeight: "300px" }}
                    />
                  </div>
                )}
              </div>
              <div className="card-footer">
                <button type="submit" className="btn btn-success">
                  Upload
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default ImageUpload;
