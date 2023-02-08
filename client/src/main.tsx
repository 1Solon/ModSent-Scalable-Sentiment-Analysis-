import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import VADERpage from "./pages/VADERpage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<VADERpage /> } />
      </Routes>
    </BrowserRouter>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
