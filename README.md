# Posterity.gurila.tools

A tactical simulation engine that models social dynamics using the Lanchester Laws of conflict. This project is a Python 3.10+ rewrite of the legacy Java Android application "Posterity", designed to simulate the "dance" between two forces: **Bison** (Active/High Morale) and **Cattle** (Passive/Low Morale).

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Copyright (C) 2026 Jefferson Richards <jefferson@richards.plus>

## Requirements

- **Python 3.10 or later** (required for modern type hinting and performance optimizations)
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- pytest >= 6.0.0 (for testing)

## Project Overview

`posterity.gurila.tools` implements a sophisticated tactical simulation system that combines:

1. **Lanchester Laws**: Mathematical models for attrition over time between opposing forces
2. **Stochastic Coefficients**: Dynamic effectiveness constants derived from Markov Chain transitions
3. **Markov Chain Dynamics**: State transitions that pull values from specific probability distributions (Normal, Left-Skewed, Right-Skewed)
4. **Tactical Heuristics**: The 3-7-12 decision framework for social approach strategies

## Core Concepts

### The Dance
The simulation models a "dance" between two population types:
- **Bison**: Represents active, high-morale individuals
- **Cattle**: Represents passive, low-morale individuals

### Input Parameters
- **Flux**: Rate of change in the system
- **Heat**: Volatility of the Markov chain (affects coefficient variation)
- **Pace**: Speed of iteration (temporal resolution)
- **Count**: Initial population sizes

### Output Analysis
The system provides tactical advice based on simulation outcomes:
- **Active vs Passive**: Approach strategy recommendation
- **3-7-12 Framework**: Target group size for optimal engagement

## Architecture

The project follows a modular Python architecture with the following key components:

### Core Engine (`core/`)
- `physics.py`: Mathematical engine implementing Lanchester equations and Markov chains
- `simulation.py`: Main simulation runner with termination conditions

### Analysis Layer (`analysis/`)
- `tactics.py`: Heuristic interpretation layer implementing 3-7-12 logic

### Interfaces (`interfaces/`)
- `vision_hooks.py`: Future AR integration hooks for computer vision input

### CLI & Main
- `main.py`: Command-line interface for running simulations

## Key Features

### Mathematical Rigor
- Vectorized operations using NumPy for efficiency
- Stable numeric solvers for differential equations
- Proper handling of stochastic processes

### Termination Conditions
The simulation runs until one of these conditions is met:
1. **Annihilation**: One population drops to ≤ 0
2. **Crossover**: Population curves cross once
3. **Double Crossover**: Curves cross twice (complex "dance")
4. **Time Limit**: Equivalent of 1.5 hours in simulation time

### Tactical Intelligence
- Interprets simulation volatility patterns
- Maps outcomes to strategic recommendations
- Provides actionable advice for social dynamics

## Technology Stack

- **Python 3.10+**: Core language with modern type hinting and performance optimizations
- **NumPy**: Vectorized mathematical operations for efficiency
- **SciPy**: Statistical distributions and advanced mathematical functions
- **Type Hints**: Full type annotation for maintainability and IDE support
- **pytest**: Comprehensive testing framework

## Contact & Support

- **Author**: Jefferson Richards
- **Email**: jefferson@richards.plus
- **License**: GNU General Public License v3.0 or later

For bug reports, feature requests, or contributions, please contact jefferson@richards.plus.

## Future AR Integration

The system is designed with hooks for future augmented reality integration:
- Computer vision input for real-time crowd analysis
- Automatic parameter extraction from video feeds
- Anomaly detection for identifying Bison vs Cattle behavior patterns

## Installation & Usage

### Prerequisites
Ensure you have Python 3.10 or later installed:
```bash
python3 --version  # Should show 3.10.x or higher
```

### Install Dependencies
```bash
# Install required packages
pip3 install numpy scipy pytest

# Or install from requirements file
pip3 install -r requirements.txt
```

### Run Simulation
```bash
# Basic usage
python3 main.py --flux 0.5 --heat 0.8 --count 50

# Grocery store scenario (should output passive recommendation)
python3 main.py --grocery-store

# JSON output for integration
python3 main.py --flux 0.2 --heat 0.9 --count 100 --json

# Quiet mode (just the recommendation)
python3 main.py --flux 0.5 --heat 0.8 --count 50 --quiet
# Expected output: "Approach as Active, Target Group 7"
```

### Run Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test module
python3 -m pytest tests/test_physics.py -v
```

## Development Philosophy

This rewrite prioritizes:
- **Efficiency**: Optimized for mobile/AR hardware deployment
- **Modularity**: Clean separation of concerns
- **Extensibility**: Ready for future AR and ML integration
- **Maintainability**: Comprehensive type hints and documentation

---

*Posterity.gurila.tools - Where tactical simulation meets social dynamics*