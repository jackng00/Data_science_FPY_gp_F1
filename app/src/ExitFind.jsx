import React, { useEffect,useState  } from 'react';
import axios from 'axios';
import './ExitFind.css';

const ExitFind = () => {
  const [displayOpen, setDisplayOpen] = useState(false);
  const [isDisabled, setIsDisabled] = useState(false);
  const [startMove, setStartMove] = useState(false);

  const [showImages, setShowImages] = useState(false);
  const [imagePath1, showImages1] = useState('');
  const [imagePath2, showImages2] = useState('');
  const [imagePath3, showImages3] = useState('');
  const [imagePath4, showImages4] = useState('');

  const [showResult, setShowResult] = useState(false);
  const [imagePath5, showImages5] = useState('');
  const [imagePath6, showImages6] = useState('');

  const handleToggleDisplay = () => {
    setDisplayOpen(!displayOpen);
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get('http://localhost:5000/auto_move_status');
        console.log(response.data);
        if (response.data.status == 1){
          showImages1(response.data.path);
          setShowImages(true)
        }
        if (response.data.status == 2){
          showImages2(response.data.path);
          setShowImages(true)
        }
        if (response.data.status == 3){
          showImages3(response.data.path);
          setShowImages(true)
        }
        if (response.data.status == 4){
          showImages4(response.data.path);
          setShowImages(true)
        }
        if (response.data.status == 5){
          showImages5(response.data.path1)
          showImages6(response.data.path2)
          setShowResult(true)
        }

      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    fetchStatus(); // Initial call
    const interval = setInterval(fetchStatus, 3000); // Call every 5 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, []);


  const startAutoMove = async () => {
    try {
      setStartMove(true)
      console.log('auto_move_new')
      axios.get('http://localhost:5000/exit_finding')
      .then(response => console.log(response.data))
      .catch(error => console.error(error));

      axios.get('http://localhost:5000/run_SLAM')
      .then(response => console.log(response.data))
      .catch(error => console.error(error));
    } catch (error) {
        console.error('Error :', error);
    }

};

  return (
    <div>
      <div className="upload-container2">
      {!startMove ?
        <button className="toggle-button" onClick={startAutoMove}>Start ExitFind</button> : 
        <div className="auto_move_font">Auto Moving</div>
      }
        
      </div>
      <div className="rounded-container" style={{border: ' 5px solid red'}}>
      <div className="display-container">
        <button className="toggle-button"  style={{ backgroundColor: displayOpen ? '#28a745' : '#dc3545' }} onClick={handleToggleDisplay}>
            {displayOpen ? 'Close Display' : 'Open Display'}
        </button>

      </div>
      
      <div className="drone-view-container">

        <div className="drone-view">
          {displayOpen && (
            <img  src="http://localhost:5000/video_feed_auto_move" width="600" height="360"/>
          )}
        </div>
        <div className="image-grid-container">
          {showImages && (
                    <div className="image-grid">
                      {imagePath1 && (  <img src={`http://localhost:5000/get_image/${imagePath1}`} alt="View 1" />)}
                      {imagePath2 && (  <img src={`http://localhost:5000/get_image/${imagePath2}`} alt="View 2" />)}
                      {imagePath3 && (  <img src={`http://localhost:5000/get_image/${imagePath3}`} alt="View 3" />)}
                      {imagePath4 && (  <img src={`http://localhost:5000/get_image/${imagePath4}`} alt="View 4" />)}
                    </div>
                )}

            {showResult && (
                    <div className="image-grid">
                      {showResult && (  <img src={`http://localhost:5000/get_image/${imagePath5}`} alt="Result 4 dimensions" />)}
                      {showResult && (  <img src={`http://localhost:5000/get_image/${imagePath6}`} alt="Merged Result" />)}
                    </div>
                )}
            </div>
          
    </div>

    </div>

    </div>

  );

}

export default ExitFind;
