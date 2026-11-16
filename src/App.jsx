
import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import VictimUI from './pages/VictimUI';
import CommandCenter from './pages/CommandCenter';
import RescueTeam from './pages/RescueTeam';
import SafeUI from './pages/SafeUI';
import { testFirebaseConnection } from './firebaseTest';

export default function App() {
  useEffect(() => {
    testFirebaseConnection();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<VictimUI />} />
        <Route path="/command-center" element={<CommandCenter />} />
        <Route path="/rescue" element={<RescueTeam />} />
        <Route path="/safe" element={<SafeUI />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
