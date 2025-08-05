import React from "react";
import Header from "./components/Header";
import MainControls from "./components/MainControls";
import DataPanel from "./components/DataPanel";
import NetworkBuilder from "./components/NetworkBuilder";
import OutputPanel from "./components/OutputPanel";
import Footer from "./components/Footer";

import "./App.css";

function App() {
  return (
    <div className="app-root">
      <Header />
      <MainControls />
      <div className="main-layout">
        <DataPanel />
        <NetworkBuilder />
        <OutputPanel />
      </div>
      <Footer />
    </div>
  );
}

export default App;