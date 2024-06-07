import React, { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";

import Nav from "./components/Nav";
import Home from "./pages/Home";
import "./App.css";
import Water1 from "./pages/Water1";
import Water2 from "./pages/Water2";
import Water3 from "./pages/Water3";

function App() {
  const [width, setWidth] = useState(window.innerWidth);

  const handleResize = () => {
    setWidth(window.innerWidth);
  };

  useEffect(() => {
    window.addEventListener("resize", handleResize);
    return () => {
      // cleanup
      window.removeEventListener("resize", handleResize);
    };
  }, []);
  return (
    <div className="App">
      <Routes>
        <Route exact path="/" Component={Home} />
        <Route path="/water1" Component={Water1} />
        <Route path="/water2" Component={Water2} />
        <Route path="/water3" Component={Water3} />
      </Routes>
      <Nav />
    </div>
  );
}

export default App;
