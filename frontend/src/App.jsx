import { Navigate, Route, Routes } from "react-router-dom";

import MainLayout from "./MainLayout";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Login from "./pages/Login";
import Settings from "./pages/Settings";
import Signal from "./pages/Signal";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />

      {/* Protected Routes */}
      <Route
        path="/dashboard"
        element={
          <MainLayout>
            <Dashboard />
          </MainLayout>
        }
      />
      <Route
        path="/signal"
        element={
          <MainLayout>
            <Signal />
          </MainLayout>
        }
      />
      <Route
        path="/history"
        element={
          <MainLayout>
            <History />
          </MainLayout>
        }
      />
      <Route
        path="/settings"
        element={
          <MainLayout>
            <Settings />
          </MainLayout>
        }
      />

      {/* Add other routes similarly */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
