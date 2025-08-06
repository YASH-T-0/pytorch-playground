# PyTorch Neural Network Playground

**PyTorch Neural Network Playground** is an open-source interactive tool for experimenting with artificial neural networks (ANNs) built on top of PyTorch.  
Inspired by classic playgrounds, our goal is to make an educational, fun, and modern environment for visualizing and training feedforward neural networks—**but with a PyTorch backend**.

---

## Project Preview


<img width="1920" height="895" alt="typeone" src="https://github.com/user-attachments/assets/5686ed7d-39b3-4f99-836c-35dce5dd14d6" />


_We are actively improving the UI!_

---

## 🚧 Project Status

- **Alpha:** The backend is fully functional using Python, Flask, and PyTorch.
- **Frontend:** A web-based UI is available for configuring the network and viewing results, but the design is still a work in progress and open to contributions!
- **Open Source:** Contributions, suggestions, and UI pull requests are welcome!

---

## Features

- **PyTorch-powered:** Train real models in Python, not just in JavaScript.
- **Flexible Architecture:** Configure layers, neurons, learning rate, and more.
- **Interactive Playground:** Adjust parameters and visualize results instantly.
- **Educational:** Visualize loss, accuracy, and decision boundaries (in progress).
- **Open Source:** Community-driven, MIT-licensed.

---

## Quickstart

### 1. Backend (Python + PyTorch)

```bash
git clone https://YASH-T-0/pytorch-playground.git
cd pytorch-playground 
pip install flask flask-cors torch numpy
python app.py

```

The backend will run on [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Frontend

- Open `index.html` in your browser (double-click or use a local web server).
- The UI lets you set layers, neurons, epochs, learning rate, etc.
- Click **Play** to train the network—results are fetched from the backend.

---

## Why PyTorch?

Most neural network playgrounds use TensorFlow.js or other JS-only libraries.  
**This playground uses real PyTorch models on the backend**—allowing for more realistic experiments, easier extension, and the ability to use real Python ML tooling.

---

## Roadmap

- [x] PyTorch backend with Flask API
- [x] Basic responsive frontend UI
- [x] SVG neural network visualization (circular/curved neurons)
- [ ] Improved frontend design (WIP)
- [ ] Upload and visualize custom datasets
- [ ] Export/import trained models
- [ ] More activation/regularization options
- [ ] Multi-class and regression support
- [ ] Community-contributed UI themes

---

## Contributing

We **welcome contributions!**

- If you’re good with design, help us improve the frontend!
- If you love PyTorch, suggest backend features, new visualizations, or optimizations.
- Please open issues for bugs, feature requests, or questions.

---

## License

MIT License.  
See [LICENSE](LICENSE).

---

## Credits

- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
- UI inspired by TensorFlow Playground and community feedback.

---

> **Note:** This project is under active development.  
> Frontend design is not final—help us make it better!
