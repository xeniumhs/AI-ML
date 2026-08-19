import React from 'react'
import { useEffect, useState } from "react";

const SystemStatus = () => {
    const [status, setStatus] = useState(null);
    const [error, setError] = useState(null);


  useEffect(() => {
    fetch("/health/")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend returned an error");
        }
        return response.json();
      })
      .then((data) => {
        setStatus(data);
      })
      .catch((err) => {
        setError(err.message);
      });
  }, []);

  if (error) {
    return (
      <div>
        <h2>System Status</h2>
        <p>🔴 Backend: Disconnected</p>
        <p>{error}</p>
      </div>
    );
  }

  if (!status) {
    return <p>Checking system...</p>;
    }
    
  return (
    <div>
      <h2>System Status</h2>

      <p>🟢 React: Running</p>
      <p>🟢 Django: {status.backend}</p>
      <p>🟢 Database: {status.database}</p>
      <p>🟢 pgvector: {status.pgvector}</p>

      <h3>🚀 System Ready</h3>
    </div>
  );
}

export default SystemStatus
