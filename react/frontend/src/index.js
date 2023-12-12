import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, createBrowserRouter } from 'react-router-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import ImageUpload1 from '../src/components/ImageUpload1';
import Login  from './components/Login';

const root = ReactDOM.createRoot(document.getElementById('root'));
const router = createBrowserRouter([
  {
    path: "/",
    element: <ImageUpload1 />,
  },
]);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
