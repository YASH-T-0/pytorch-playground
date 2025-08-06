// Theme toggle
const themeBtn = document.getElementById('toggle-theme');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
function setTheme(mode) {
  document.body.setAttribute('data-theme', mode);
  themeBtn.textContent = mode === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('theme', mode);
}
themeBtn.addEventListener('click', () => {
  const current = document.body.getAttribute('data-theme') || (prefersDark ? 'dark' : 'light');
  setTheme(current === 'dark' ? 'light' : 'dark');
});
window.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('theme');
  setTheme(saved || (prefersDark ? 'dark' : 'light'));
});

// Features dropdown logic
document.getElementById('features-toggle').onclick = function() {
  document.getElementById('features-panel').classList.toggle('hidden');
};
window.addEventListener('click', function(e) {
  const panel = document.getElementById('features-panel');
  const btn = document.getElementById('features-toggle');
  if (!btn.contains(e.target) && !panel.contains(e.target)) {
    panel.classList.add('hidden');
  }
});

// Font size slider
const fontSlider = document.getElementById('font-size-slider');
const fontVal = document.getElementById('font-size-value');
fontSlider.addEventListener('input', () => {
  document.body.style.fontSize = fontSlider.value + 'px';
  fontVal.textContent = fontSlider.value + 'px';
});

// Dummy output canvas
const canvas = document.getElementById('output-canvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  for (let i = 0; i < 100; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = (x > canvas.width / 2) ? "#ff9800" : "#2196f3";
    ctx.fill();
  }
}

// Dummy loss chart
function drawLossChart(losses) {
  const chart = document.getElementById('loss-chart');
  if (!chart) return;
  const ctx = chart.getContext('2d');
  ctx.clearRect(0,0,chart.width,chart.height);
  ctx.strokeStyle = "#2196f3";
  ctx.beginPath();
  ctx.moveTo(0, chart.height-10);
  for (let i = 1; i < 20; i++) {
    ctx.lineTo(i * 9, chart.height-10 - Math.random() * 35);
  }
  ctx.stroke();
}
drawLossChart();

// Dummy accuracy bar
function setAccuracyBar(acc) {
  let bar = document.getElementById("accuracy-bar-inner");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "accuracy-bar-inner";
    document.getElementById("accuracy-bar").appendChild(bar);
  }
  bar.style.width = Math.round(acc * 100) + "%";
}
setAccuracyBar(0.72);

// Layers/neurons controls
let layers = [
  { neurons: 4 }, { neurons: 3 }, { neurons: 2 }, { neurons: 2 }, { neurons: 2 }
];
function updateLayerUI() {
  const configs = document.getElementById("layer-configs");
  configs.innerHTML = "";
  layers.forEach((layer, idx) => {
    const div = document.createElement("div");
    div.className = "layer-config";
    div.innerHTML = `Layer ${idx+1}: <input type="number" value="${layer.neurons}" min="1" max="12" style="width:40px"> neurons`;
    div.querySelector("input").addEventListener("input", e => {
      layers[idx].neurons = parseInt(e.target.value, 10);
      drawNetworkSVG();
    });
    configs.appendChild(div);
  });
  document.getElementById("layer-count").textContent = `${layers.length} Layers`;
  drawNetworkSVG();
}
document.getElementById("add-layer").onclick = function() {
  layers.push({ neurons: 2 });
  updateLayerUI();
};
document.getElementById("remove-layer").onclick = function() {
  if (layers.length > 1) layers.pop();
  updateLayerUI();
};

// --- NEURAL NETWORK SVG VISUALIZATION (Circular Neurons, Responsive) ---
function drawNetworkSVG() {
  const svg = document.getElementById("network-svg");
  svg.innerHTML = '';
  // Make the SVG as wide as its container (network-visual), but max 98% of panel
  const width = svg.parentElement.offsetWidth > 350 ? svg.parentElement.offsetWidth : 350;
  svg.setAttribute('width', width);
  svg.setAttribute('height', 180);

  // Spacing
  const layerCount = layers.length;
  const maxNeurons = Math.max(...layers.map(l => l.neurons), 2);
  const layerGap = width / (layerCount + 2);
  const neuronGap = 28;
  const neuronRadius = 13;

  // Input layer (2 neurons)
  let prevCenters = [];
  for (let i = 0; i < 2; i++) {
    const cx = layerGap * 0.6;
    const cy = 55 + i * neuronGap;
    prevCenters.push([cx, cy]);
    const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circ.setAttribute("cx", cx);
    circ.setAttribute("cy", cy);
    circ.setAttribute("r", neuronRadius);
    circ.setAttribute("fill", "#5ad4ff");
    circ.setAttribute("stroke", "#fff");
    circ.setAttribute("stroke-width", "2");
    svg.appendChild(circ);
  }

  // Hidden layers
  let lastCenters = prevCenters;
  layers.forEach((layer, lidx) => {
    let centers = [];
    const cx = layerGap * (lidx + 1.6);
    const startY = 55 + (maxNeurons - layer.neurons) * neuronGap / 2;
    for (let n = 0; n < layer.neurons; n++) {
      const cy = startY + n * neuronGap;
      centers.push([cx, cy]);
      // Neuron
      const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circ.setAttribute("cx", cx);
      circ.setAttribute("cy", cy);
      circ.setAttribute("r", neuronRadius);
      circ.setAttribute("fill", "#223040");
      circ.setAttribute("stroke", "#5ad4ff");
      circ.setAttribute("stroke-width", "2");
      svg.appendChild(circ);

      // Curved Connections
      lastCenters.forEach(([pcx, pcy]) => {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const mx = pcx + neuronRadius, my = pcy;
        const ex = cx - neuronRadius, ey = cy;
        const c1x = mx + (layerGap/3), c1y = my;
        const c2x = ex - (layerGap/3), c2y = ey;
        path.setAttribute("d", `M${mx},${my} C${c1x},${c1y} ${c2x},${c2y} ${ex},${ey}`);
        path.setAttribute("stroke", "#5ad4ff");
        path.setAttribute("stroke-width", "1.5");
        path.setAttribute("fill", "none");
        svg.appendChild(path);
      });
    }
    lastCenters = centers;
  });

  // Output layer (2 neurons, classification)
  const outCx = layerGap * (layerCount + 1.6);
  for (let i = 0; i < 2; i++) {
    const cy = 70 + i * neuronGap;
    const circ = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circ.setAttribute("cx", outCx);
    circ.setAttribute("cy", cy);
    circ.setAttribute("r", neuronRadius);
    circ.setAttribute("fill", "#5ad4ff");
    circ.setAttribute("stroke", "#fff");
    circ.setAttribute("stroke-width", "2");
    svg.appendChild(circ);

    // Connect last hidden layer to output (curved)
    lastCenters.forEach(([pcx, pcy]) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const mx = pcx + neuronRadius, my = pcy;
      const ex = outCx - neuronRadius, ey = cy;
      const c1x = mx + (layerGap/3), c1y = my;
      const c2x = ex - (layerGap/3), c2y = ey;
      path.setAttribute("d", `M${mx},${my} C${c1x},${c1y} ${c2x},${c2y} ${ex},${ey}`);
      path.setAttribute("stroke", "#5ad4ff");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("fill", "none");
      svg.appendChild(path);
    });
  }
}
updateLayerUI();

// Dummy run history
const history = [
  { epoch: 1000, loss: 0.55, acc: 0.71 },
  { epoch: 2000, loss: 0.51, acc: 0.74 },
  { epoch: 3000, loss: 0.48, acc: 0.76 }
];
const runHistoryList = document.getElementById("run-history");
history.forEach(run => {
  const li = document.createElement("li");
  li.textContent = `Epoch ${run.epoch}: Loss ${run.loss.toFixed(2)}, Acc ${(run.acc*100).toFixed(1)}%`;
  runHistoryList.appendChild(li);
});

// Tooltips (for accessibility and help)
document.querySelectorAll('.tooltip').forEach(elem => {
  elem.addEventListener('mouseenter', function(e) {
    const popup = document.getElementById('tooltip-popup');
    popup.textContent = elem.title;
    popup.style.display = 'block';
    popup.style.left = (e.clientX + 12) + 'px';
    popup.style.top = (e.clientY + 12) + 'px';
  });
  elem.addEventListener('mouseleave', function() {
    document.getElementById('tooltip-popup').style.display = 'none';
  });
});

// Help button popup (guided walkthrough, dummy)
document.getElementById("help-btn").onclick = function() {
  alert("Welcome to the Neural Network Playground!\n\n- Set your data and features\n- Adjust model layers and neurons\n- Tune parameters\n- See results and history\n\nAll in the browser. For real training, connect to TensorFlow.js or Python backend!");
};

// Make SVG responsive on resize
window.addEventListener("resize", drawNetworkSVG);

function trainNetwork(layers, epochs, lr, data) {
  fetch('http://127.0.0.1:5000/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      layers: layers,
      epochs: epochs,
      lr: lr,
      data: data
    })
  })
  .then(resp => resp.json())
  .then(res => {
    document.getElementById('test-loss').textContent = res.final_loss.toFixed(3);
    document.getElementById('train-loss').textContent = res.final_loss.toFixed(3);
    setAccuracyBar(res.accuracy);
    drawLossChart(res.losses);
    // Optional: draw predictions on output canvas!
  });
}

// document.querySelector('.play').onclick = function() {
//   // Get layers, epochs, lr from UI
//   const layers = [4, 3, 2]; // Replace with UI values!
//   const epochs = 200;
//   const lr = 0.01;
//   // Send dummy data for demo
//   trainNetwork(layers, epochs, lr, null);
// };






document.querySelector('.play').onclick = function() {
  const modelType = document.getElementById('model-type-select').value;
  const epochs = 200;
  const lr = 0.01;
  let layers = [4, 3, 2]; // replace with user UI if needed

  // Prepare example data for each model type
  let data = null;
  if (modelType === "ANN") {
    // Shape: X: (N, features), y: (N,)
    data = {
      X: Array.from({length: 200}, () => [Math.random()*2-1, Math.random()*2-1]),
      y: Array.from({length: 200}, (_,i) => Math.random()>0.5 ? 1 : 0)
    };
  } else if (modelType === "CNN") {
    // Shape: X: (N, 1, 28, 28), y: (N,)
    data = {
      X: Array.from({length: 32}, () =>
        [Array.from({length: 28}, () =>
          Array.from({length: 28}, () => Math.random())
        )]
      ),
      y: Array.from({length: 32}, () => Math.random() > 0.5 ? 1 : 0)
    };
  } else if (modelType === "RNN") {
    // Shape: X: (N, seq_len, feature), y: (N,)
    data = {
      X: Array.from({length: 32}, () =>
        Array.from({length: 10}, () =>
          Array.from({length: 4}, () => Math.random())
        )
      ),
      y: Array.from({length: 32}, () => Math.random() > 0.5 ? 1 : 0)
    };
  }

  fetch('http://127.0.0.1:5000/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_type: modelType,
      layers: layers,
      epochs: epochs,
      lr: lr,
      data: data
    })
  })
    .then(resp => resp.json())
    .then(res => {
      document.getElementById('test-loss').textContent = res.final_loss.toFixed(3);
      document.getElementById('train-loss').textContent = res.final_loss.toFixed(3);
      setAccuracyBar(res.accuracy);
      drawLossChart(res.losses);
    });
};