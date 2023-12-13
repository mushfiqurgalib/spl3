import './App.css';
// import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
// import React from 'react';

import { BrowserRouter as Router, Route, Routes, BrowserRouter, HashRouter } from 'react-router-dom';
import ImageUpload1 from "./components/ImageUpload1";
import Login from './components/Login';
import Signup from './components/Signup';
import History from './components/History';

function App() {
  return (
    <BrowserRouter>
        <Routes>
          <Route path='/' element ={<ImageUpload1/>}/>
          <Route path='/login' element ={<Login/>}/>
          <Route path='/signup' element ={<Signup/>}/>
          <Route path='/history' element ={<History/>}/>
        </Routes>
    </BrowserRouter>
  );
}

export default App;
