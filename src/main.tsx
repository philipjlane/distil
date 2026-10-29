import React from "react";
import ReactDOM from "react-dom/client";
import { getCurrentWindow } from "@tauri-apps/api/window";
import App from "./App";
import { LivePanel } from "./LivePanel";
import { TrayPanel } from "./TrayPanel";

// One page serves every window; the label says which UI it gets.
const label = getCurrentWindow().label;

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {label === "tray" ? <TrayPanel /> : label === "live" ? <LivePanel /> : <App />}
  </React.StrictMode>,
);
