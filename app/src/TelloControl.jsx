import React, { useEffect,useState  } from 'react';
import axios from 'axios';
import './TelloControl.css';

const TelloControl = () => {
  const [displayOpen, setDisplayOpen] = useState(false);
  const [detectOpen, setDetectOpen] = useState(false);
  const [imageSrc, setImageSrc] = useState('');
  const [imagePath, setImagePath] = useState(null);
  const [summary, setSummary] = useState('');

    const handleToggleDisplay = () => {
        setDisplayOpen(!displayOpen);
    };

    const handleToggleDetect = () => {
      setDetectOpen(!detectOpen);
      axios.get(`http://localhost:5000/change_normal_predict_state`)
        .then(response => console.log(response.data))
        .catch(error => console.error(error));
  };


  const handleKeyPress = (event) => {
    if (event.key === 'z') {
      handleTakeoff();
    }
    if (event.key === 'x') {
      handleLand();
    }
    if (event.key === 'w') {
      handleMove('forward', 20);
    }
    if (event.key === 's') {
      handleMove('backward', 20);
    }
    if (event.key === 'a') {
      handleMove('left', 20);
    }
    if (event.key === 'd') {
      handleMove('right', 20);
    }
    if (event.key === 'o') {
      handleRotate('counter-clockwise', 45);
    }
    if (event.key === 'p') {
      handleRotate('clockwise', 45);
    }
  };

  const fetchImage = async () => {
    try {
      const response = await fetch('http://localhost:5000/one_time_predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      setImagePath(data.image_path);
      setSummary(data.result)
      console.log(data)
    } catch (error) {
      console.error('Error fetching the image:', error);
    }
  };

  useEffect(() => {
    document.addEventListener('keypress', handleKeyPress);

    // Clean up the event listener
    return () => {
        document.removeEventListener('keypress', handleKeyPress);
    };
  }, []);

    const handleTakeoff = () => {
      console.log('takeff')
        axios.get('http://localhost:5000/takeoff')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleLand = () => {
      console.log('land')
        axios.get('http://localhost:5000/land')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleMove = (direction, distance) => {
        axios.get(`http://localhost:5000/move?direction=${direction}&distance=${distance}`)
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleRotate = (direction, degree) => {
      axios.get(`http://localhost:5000/rotate?direction=${direction}&degree=${degree}`)
          .then(response => console.log(response.data))
          .catch(error => console.error(error));
  };

    return (
      <div className="rounded-container" style={{border: ' 5px solid #007bff'}}>
      <div className="display-container">
        <button className="toggle-button"  style={{ backgroundColor: displayOpen ? '#28a745' : '#dc3545' }} onClick={handleToggleDisplay}>
            {displayOpen ? 'Close Display' : 'Open Display'}
        </button>

        <button className="toggle-button"  style={{ backgroundColor: 'blue' }} onClick={fetchImage}>Get Image</button>
        

      <button className="toggle-button"  style={{ backgroundColor: detectOpen ? '#28a745' : '#dc3545' }} onClick={handleToggleDetect}>
            {detectOpen ? 'Close Detect' : 'Open Detect'}
        </button>
      </div>

      <div className="display-container">
      {imagePath && <img src={`http://localhost:5000/get_image/${imagePath}`} alt="Predicted" width="600" height="360"/>}
      {imagePath &&<div className="summary-container"> <h3>Object Class Summary</h3> <p>{summary}</p></div>}
      </div>
      
      <div className="drone-view-container">
        
        
        <div className="controller-left">
        <button className="round-button" onClick={() => handleMove('forward', 20)}>↑</button>
        <button className="round-button" onClick={() => handleMove('backward', 20)}>↓</button>
        <button className="round-button" onClick={() => handleMove('left', 20)}>←</button>
        <button className="round-button" onClick={() => handleMove('right', 20)}>→</button>
        
        </div>
        <div className="drone-view">
          {displayOpen && (
            <img  src="http://localhost:5000/video_feed" width="600" height="360"/>
          )}
        </div>
        <div className="controller-right">
        <button className="round-button" onClick={() =>handleTakeoff()}>T</button>
        <button className="round-button" onClick={() =>handleLand()}>L</button>
        <button className="round-button" onClick={() => handleRotate('counter-clockwise',45)}>↺</button>
        <button className="round-button" onClick={() => handleRotate('clockwise',45)}> ↻</button>
        <button className="round-button" onClick={() => handleMove('up', 20)}>Up</button>
        <button className="round-button" onClick={() => handleMove('down', 20)}>Dw</button>
        
        </div>

    </div>

    </div>
    );
};

export default TelloControl;