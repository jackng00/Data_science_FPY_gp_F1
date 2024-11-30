import React, { useEffect,useRef } from 'react';

const WebSocketComponent = () => {

    const socket = new WebSocket('ws://localhost:8765');
    const videoRef = useRef();

    socket.onopen = () => {
      console.log('WebSocket connection established');
      socket.send('established');
    };

    socket.onmessage = (event) => {
      console.log('Message received from server:', event.data);

      const frame = event.data;
      if (videoRef.current) {
        console.log('frame');
        videoRef.current.src = `data:image/jpeg;base64,${frame}`;
      }
    };

    // Send a message to the WebSocket server
    const sendMessage = (message) => {
      console.log('Message ', message);
      socket.send(message);
    };


  return <div>
      <div>WebSocket Component</div>
      <video ref={videoRef} controls autoPlay />
      <button onClick={() => sendMessage('takeoff')}>Take Off</button>
      <button onClick={() => sendMessage('land')}>Land</button>
      <button onClick={() => sendMessage('move_forward 20')}>Move Forward</button>
      </div>
      ;
};

export default WebSocketComponent;