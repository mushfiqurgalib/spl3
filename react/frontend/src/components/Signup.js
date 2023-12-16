import React, { useState } from 'react';
import axios from 'axios';
import './style.css';
import { useNavigate } from 'react-router-dom';


const Signup = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');  
  const [password, setPassword] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault(); // Prevents the default form submission behavior

    try {
      console.log(username, password);

      const response = await axios.post('http://127.0.0.1:5000/signup', {
        username,
        email,
        password,
      });

      if (response.status === 200) {
        const data = response.data;
        console.log(data.message);
      

        // Show success alert
        alert('Signup successful');
        navigate('/');
    } else if(response.status === 400) { 
        // Show unsuccessful alert
        alert('Signup unsuccessful');
    }
    else{
      alert('Signup unsuccessful');
    }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className='auth-body'>
    <div className='auth-wrapper'>
      <div className='auth-inner'>
        <form onSubmit={handleLogin}>
          <h3>Sign Up</h3>
          <div className="mb-3">
            <label>Username</label>
            <input
              type="text"
              className="form-control"
              placeholder="Enter Name"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="mb-3">
            <label>Email</label>
            <input
              type="email"
              className="form-control"
              placeholder="Enter Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="mb-3">
            <label>Password</label>
            <input
              type="password"
              className="form-control"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="mb-3">
            <div className="custom-control custom-checkbox">
              <div>
                Already have an account? <a href="/">Sign In</a>
              </div>
            </div>
          </div>
          <div className="d-grid">
            <button type="submit" className="btn btn-primary">
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
    </div>
  );
};

export default Signup;
