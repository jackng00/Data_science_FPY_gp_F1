import React, { useEffect,useState  } from 'react';
import axios from 'axios';
import './ObjectDetect.css';

const ObjectDetect = () => {
  const [displayOpen, setDisplayOpen] = useState(false);
  const [isDisabled, setIsDisabled] = useState(false);

  const [image, setImage] = useState(null);
  const [uploadfile, setFile] = useState(null);
  const [targetClass, setTargetClass] = useState("Loading...");

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    console.log('file',file)
    setFile(file)
    if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
            setImage(reader.result);
        };
        reader.readAsDataURL(file);
    }
};

const handleSubmit = async () => {
  const fd = new FormData();
  fd.append('file', uploadfile);
  console.log('fd',fd,uploadfile)
  setIsDisabled(true);
  setTargetClass("Loading...");

  try {
      const response = await axios.post('http://localhost:5000/upload_object', fd, {
          headers: {
              'Content-Type': 'multipart/form-data'
          }
      });
      console.log(response.data);
      const target_class = response.data.class_name ;
      console.log('target_class',target_class);
      setTargetClass('Target class: ' + target_class);
      

  } catch (error) {
      console.error('Error uploading image:', error);
  }

};

    const handleToggleDisplay = () => {
        setDisplayOpen(!displayOpen);
    };

  const handleKeyPress = (event) => {
    //if (!isDisabled){
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
    //}
    
  };

  useEffect(() => {
    document.addEventListener('keypress', handleKeyPress);

    // Clean up the event listener
    return () => {
        document.removeEventListener('keypress', handleKeyPress);
    };
  }, []);

    const handleTakeoff = () => {
      //if (!isDisabled){
        axios.get('http://localhost:5000/takeoff')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
      //}
    };

    const handleLand = () => {
        axios.get('http://localhost:5000/land')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleMove = (direction, distance) => {
      //if (!isDisabled){
        axios.get(`http://localhost:5000/move?direction=${direction}&distance=${distance}`)
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
      //}
    };

    const handleRotate = (direction, degree) => {
      //if (!isDisabled){
        axios.get(`http://localhost:5000/rotate?direction=${direction}&degree=${degree}`)
          .then(response => console.log(response.data))
          .catch(error => console.error(error));
      //}
  };

    return (
      <div>
      <div className="upload-container">
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        {image && (
            <div>
                <h2>Uploaded Image:</h2>
                <img src={image} alt="Uploaded" className="uploaded-image" />
            </div>
        )}
        <button onClick={handleSubmit}>Upload Image</button>
        
        {isDisabled && <p>{targetClass}</p>}
    </div>
      <div className="rounded-container" style={{border: ' 5px solid green'}}>
      <div className="display-container">
        <button className="toggle-button"  style={{ backgroundColor: displayOpen ? '#28a745' : '#dc3545' }} onClick={handleToggleDisplay}>
            {displayOpen ? 'Close Display' : 'Open Display'}
        </button>
        
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
            <img  src="http://localhost:5000/video_feed_object" width="600" height="360"/>
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

    </div>
    );
};

export default ObjectDetect;