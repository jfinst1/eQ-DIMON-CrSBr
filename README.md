# eQ-DIMON: Quantum-Classical CrSBr Simulator

Welcome to **eQ-DIMON**—a mind-bending hybrid neural network that fuses classical deep learning with a 6-qubit quantum circuit to solve Poisson’s equation on CrSBr (chromium sulfide bromide) domains. This isn’t just another PDE solver—it’s a quantum-powered beast simulating electrostatic potentials in a 2D magnetic semiconductor, complete with strain-adaptive grids, magnetic vector potentials, and enough entanglement to make a quantum physicist say, *“What the heck!?”*

## What’s This About?

CrSBr is a 2D van der Waals material with anisotropic orthorhombic structure, tunable antiferromagnetism, and a bandgap of ~1.5–1.8 eV. This project models its steady-state electrostatic potential under strain, voltage boundaries, and magnetically influenced charge distributions. Using PyTorch and PennyLane, eQ-DIMON predicts 64×64 potential maps, blending:

- **Classical MIONet**: Multi-input operator network with branches for strain (θ), boundary conditions (bc), and charge density (ρ).
- **6-Qubit Quantum Circuit**: Three variational layers encoding physical parameters, modulating the solution with quantum flair.
- **Physics-Informed Loss**: Enforces Poisson’s equation (∇²u = -ρ/ε) with magnetic vector potential coupling.

Think of it as a playground for CrSBr’s magneto-electric quirks—fast, robust, and ready for real quantum hardware.

## Features

- **Adaptive Grid Sampling**: Dynamically refines the mesh where strain gradients peak, capturing CrSBr’s deformations.
- **Quantum Boost**: A 6-qubit circuit with RY, RX, RZ gates, CNOTs, and CZs—18 trainable parameters tweaking the potential.
- **Magnetic Influence**: Vector potential (A-field) ties CrSBr’s antiferromagnetic order to the electric field.
- **Robustness**: Error handling, gradient clipping, and detailed logging keep it stable.
- **Visualization**: Contour plots with magnetic streamlines—science meets art.

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/eq-dimon-crbr.git
cd eq-dimon-crbr
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch (`torch>=1.10`)
- PennyLane (`pennylane>=0.28`)
- NumPy, SciPy, Matplotlib
- Multiprocessing support (standard in Python)

Create a `requirements.txt` with:

```
torch>=1.10
pennylane>=0.28
numpy
scipy
matplotlib
```

## Usage

Run the main script to generate data, train the model, and visualize results:

```bash
python eq_dimon_crbr.py
```

- **Data Generation**: Creates 300 CrSBr samples with random strain (0–3%), boundary voltages (-1 to 1 V), and magnetic charge distributions.
- **Training**: 10 epochs (tweakable) with a batch size of 64, early stopping, and learning rate scheduling.
- **Testing**: Predicts potentials for 5 test samples, plotting true vs. predicted fields with magnetic streamlines.

Output includes:

- Potential contour plots with overlaid magnetic field lines.
- Quantum weight evolution graph.
- Training/validation loss logs and average test MSE.

## Code Structure

`eq_dimon_crbr.py`: Main script with all components:

- `generate_crbr_domain`: Adaptive domain generation.
- `EnhancedMIONet`: Quantum-classical network.
- `eQ_DIMON`: Training and prediction logic.
- Data generation and visualization.

## Why It’s Cool

- **Quantum Edge**: A 6-qubit circuit isn’t just for show—it hints at quantum advantage for PDEs, ready for real hardware via PennyLane.
- **CrSBr Fidelity**: Captures strain (up to 3%), permittivity (ε ≈ 10), and magnetic coupling—grounded in real physics.
- **Mind-Blowing Visuals**: See how CrSBr’s magnetism shapes its potential in real-time.

## Future Directions

- **Real Quantum Hardware**: Swap the simulator for IBM Q or Rigetti via PennyLane.
- **Dynamic A-Field**: Integrate micromagnetic models for accurate magnetic fields.
- **3D Stacking**: Extend to multilayer CrSBr for 3D simulations.

## Contributing

Got ideas? Open an issue or PR! We’d love help with:

- Quantum circuit optimization.
- CrSBr experimental data integration.
- Performance tweaks for larger grids.

## License

MIT License—hack away, share, and cite if you use it!

## Acknowledgments

Inspired by CrSBr research and the wild world of quantum machine learning. Built with love, coffee, and a dash of quantum weirdness.