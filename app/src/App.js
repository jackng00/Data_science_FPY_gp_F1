import logo from './logo.svg';
import './App.css';
import WebSocketComponent from './WebSocket';
import TelloControl from './TelloControl';
import AutoDetect from './AutoDetect';
import ObjectDetect from './ObjectDetect';
import AutoMove from './AutoMove';
import React, { useEffect,useState  } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import axios from 'axios';
import ConfigPage from './ConfigPage';


const App = () => {
  const [activeTab, setActiveTab] = useState('tab1');

  const callBackendChangeState = (state) => {
    axios.get(`http://localhost:5000/change_state?state=${state}`)
            .then(response => console.log(response.data))
            .catch(error => console.error(error));

  }

  const renderTabContent = () => {
      switch(activeTab) {
          case 'tab1':
              return <TelloControl />;
          case 'tab2':
              return <AutoDetect />;
          case 'tab3':
              return <ObjectDetect />;
            case 'tab4':
              return <AutoMove />;
            case 'tab5':
                return <ConfigPage />;
          default:
              return <TelloControl />;
      }
  }

  return (
      <div>
          <div className="tabs">
              <button style={{backgroundColor: 'blue'}} onClick={() => {setActiveTab('tab1'); callBackendChangeState('normal');} }>Manual Control</button>
              <button style={{backgroundColor: 'orange'}} onClick={() => {setActiveTab('tab2'); callBackendChangeState('auto_face');} }>Auto Face Detect</button>
              <button style={{backgroundColor: 'green'}} onClick={() => {setActiveTab('tab3'); callBackendChangeState('object');} }>Objects Detect</button>
              <button style={{backgroundColor: 'red'}} onClick={() => {setActiveTab('tab4'); callBackendChangeState('auto_move');} }>Auto Move Detect</button>
              <button style={{backgroundColor: 'grey'}} onClick={() => {setActiveTab('tab5'); } }>Config</button>
          </div>
          <TransitionGroup>
              <CSSTransition
                  key={activeTab}
                  timeout={1000}
                  classNames="slide"
              >
                  <div className="tab-content">
                      {renderTabContent()}
                  </div>
              </CSSTransition>
          </TransitionGroup>
      </div>
  );
}


export default App;
