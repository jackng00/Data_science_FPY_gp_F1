import React, { useEffect,useState  } from 'react';
import axios from 'axios';
import './AutoDetect.css';

const AutoDetect = () => {
    const [displayOpen, setDisplayOpen] = useState(false);
    const [isDisabled, setIsDisabled] = useState(false);
    const [status, setStatus] = useState(false);
    const [action, setAction] = useState('');

    const [image, setImage] = useState(null);
    const [uploadfile, setFile] = useState(null);
    const [targetClass, setTargetClass] = useState("Loading...");

    useEffect(() => {
        const fetchStatus = async () => {
          try {
            const response = await axios.get('http://localhost:5000/face_detect_status');
            console.log(response.data)
            setStatus(response.data.status);
            setAction(response.data.action)
          } catch (error) {
            console.error('Error fetching status:', error);
          }
        };

        fetchStatus(); // Initial call
        const interval = setInterval(fetchStatus, 3000); // Call every 5 seconds

        return () => clearInterval(interval); // Cleanup on unmount
      }, []);

  const handleToggleDisplay = () => {
    setDisplayOpen(!displayOpen);
    };

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

    try {
        const response = await axios.post('http://localhost:5000/upload', fd, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        console.log(response.data);
        setTargetClass('Upload Finished' );

    } catch (error) {
        console.error('Error uploading image:', error);
    }

    // fetch('http://localhost:5000/upload', {
    //   method: "POST",
    //   body: fd
    // })
    // .then(res => res.json())
    // .then(data => console.log(data))
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

  useEffect(() => {
    document.addEventListener('keypress', handleKeyPress);

    // Clean up the event listener
    return () => {
        document.removeEventListener('keypress', handleKeyPress);
    };
  }, []);

    const handleTakeoff = () => {
        axios.get('http://localhost:5000/takeoff')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleTakeoff2 = () => {
        axios.get('http://localhost:5000/face_detect_takeoff')
            .then(response => console.log(response.data))
            .catch(error => console.error(error));
    };

    const handleLand = () => {
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
    <div>
      <div className="upload-container">
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        {image && (
            <div>
                <h2>Target Face:</h2>
                <img src={image} alt="Uploaded" className="uploaded-image" />
            </div>
        )}
        <button onClick={handleSubmit}>Upload Image</button>

        {isDisabled && <p>{targetClass}</p>}
    </div>
      <div className="rounded-container" style={{border: ' 5px solid orange'}}>
      <div className="display-container">
        <button className="toggle-button"  style={{ backgroundColor: displayOpen ? '#28a745' : '#dc3545' }} onClick={handleToggleDisplay}>
            {displayOpen ? 'Close Display' : 'Open Display'}
        </button>
      </div>

      <div className="display-container" style={{padding: '0px'}}>
            {action && <p className='action-word'>Flight action:{action}</p>}
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
            <img  src="http://localhost:5000/video_feed_autoface" width="600" height="360"/>
          )}
        </div>
        <div className="controller-right">
        <button className="round-button" onClick={() =>handleTakeoff2()}>T</button>
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

}

export default AutoDetect;