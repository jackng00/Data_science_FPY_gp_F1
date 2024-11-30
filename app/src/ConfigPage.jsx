import React, { useEffect,useState  } from 'react';
import axios from 'axios';
import './ConfigPage.css';

const ConfigPage = () => {
  const [models, setModels] = useState(['yolov8n.pt','yolov11n.pt']);
  const [selectedModel, setSelectedModel] = useState('');

  useEffect(() => {
      // Fetch the list of models from the backend
      // const fetchModels = async () => {
      //     const response = await axios.get('http://localhost:5000/models'); // Adjust the endpoint as needed
      //     setModels(response.data.models);
      // };

      // fetchModels();
  }, []);

  const handleModelChange = (event) => {
      setSelectedModel(event.target.value);
  };

  const handleSubmit = async () => {
      await axios.post('http://localhost:5000/select_model', { model_name: selectedModel });
      alert(`Model ${selectedModel} selected.`);
  };

  return (
      <div>
          <select value={selectedModel} onChange={handleModelChange}>
              <option value="">Select a model</option>
              {models.map((model) => (
                  <option key={model} value={model}>
                      {model}
                  </option>
              ))}
          </select>
          <button onClick={handleSubmit}>Load Model</button>
      </div>
  );
};

export default ConfigPage;