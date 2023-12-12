import './App.css';
// import React from 'react';

import { BrowserRouter as Router, Route, Routes, BrowserRouter, HashRouter } from 'react-router-dom';
import ImageUpload1 from "./components/ImageUpload1";
import Login from './components/Login';

function App() {
  return (
    <BrowserRouter>
        <Routes>
          <Route path='/' element ={<ImageUpload1/>}/>
          <Route path='/login' element ={<Login/>}/>
        </Routes>
    </BrowserRouter>
  );
}

export default App;
