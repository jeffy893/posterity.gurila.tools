# Architecture Specification - Posterity.gurila.tools

## System Overview

The `posterity.gurila.tools` system is a tactical simulation engine that models social dynamics through mathematical simulation. The architecture is designed for efficiency, modularity, and future AR integration.

**Requirements**: Python 3.10 or later for modern type hinting, performance optimizations, and advanced language features.

**License**: GNU General Public License v3.0 or later  
**Author**: Jefferson Richards <jefferson@richards.plus>

## Package Structure

```
posterity/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── physics.py          # Mathematical engine (Lanchester + Markov)
│   └── simulation.py       # Simulation runner and termination logic
├── analysis/
│   ├── __init__.py
│   └── tactics.py          # Tactical heuristics (3-7-12 framework)
├── interfaces/
│   ├── __init__.py
│   └── vision_hooks.py     # Future AR integration hooks
└── tests/
    ├── __init__.py
    ├── test_physics.py
    ├── test_simulation.py
    └── test_tactics.py
```

## Core Components

### 1. Mathematical Engine (`core/physics.py`)

**MarkovChain Class**
- Manages morale state transitions
- Supports Normal, Left-Skewed, and Right-Skewed distributions
- Provides coefficient generation for Lanchester equations

**LanchesterSolver Class**
- Implements numeric solver for Lanchester Square Law equations
- Handles dynamic coefficient updates from Markov Chain
- Uses vectorized NumPy operations for efficiency

**Key Equations:**
```
d(Bison)/dt = -beta * Cattle
d(Cattle)/dt = -alpha * Bison
```

Where `alpha` and `beta` are updated by MarkovChain at each time step.

### 2. Simulation Engine (`core/simulation.py`)

**SimulationRunner Class**
- Orchestrates the complete simulation lifecycle
- Implements termination condition checking
- Returns full trajectory history for analysis

**Input Parameters:**
- `pace`: Speed of iteration (affects dt)
- `flux`: Rate of change in the system
- `heat`: Volatility of Markov chain
- `count`: Initial population sizes

**Termination Conditions:**
1. **Annihilation**: Population ≤ 0
2. **Crossover**: Curves cross once
3. **Double Crossover**: Curves cross twice
4. **Time Limit**: 1.5 hours simulation time

### 3. Tactical Analysis (`analysis/tactics.py`)

**TacticalBrain Class**
- Interprets simulation results
- Implements 3-7-12 heuristic framework
- Provides Active/Passive recommendations

**Decision Logic:**
- High volatility (Double Crossover) → 'Longevity' strategy (Target 12)
- Dominance (Fast win) → 'Strike' strategy (Target 3)
- Stability (No crossover) → 'Balance' strategy (Target 7)

**Active/Passive Logic:**
- High Heat + Bison Win → 'Active Approach'
- Low Heat + Cattle Win → 'Passive Grazing'

### 4. Future AR Integration (`interfaces/vision_hooks.py`)

**SceneAnalyzer Abstract Class**
- Template for computer vision integration
- Hooks for real-time parameter extraction
- Anomaly detection framework for Bison/Cattle identification

## Data Flow

```
Input Parameters (Flux, Heat, Pace, Count)
    ↓
MarkovChain generates coefficients
    ↓
LanchesterSolver computes population dynamics
    ↓
SimulationRunner monitors termination conditions
    ↓
TacticalBrain analyzes results
    ↓
Output: Strategy recommendation (Active/Passive + Target Group)
```

## Performance Considerations

### Vectorization Strategy
- All mathematical operations use NumPy arrays
- Batch processing of time steps where possible
- Minimal Python loops in critical paths

### Memory Efficiency
- Streaming computation for long simulations
- Configurable history retention
- Lazy evaluation of intermediate results

### Mobile/AR Optimization
- Designed for resource-constrained environments
- Configurable precision vs speed trade-offs
- Minimal external dependencies

## Type System

Full Python 3.10+ type hints throughout for maximum IDE support and runtime safety:

```python
from typing import List, Tuple, Optional, Protocol
import numpy as np
from numpy.typing import NDArray

class SimulationResult(Protocol):
    trajectory: NDArray[np.float64]
    termination_reason: str
    recommendation: str
```

**Python 3.10+ Features Used**:
- Union types with `|` operator
- Pattern matching with `match`/`case` statements
- Improved error messages
- Performance optimizations
- Enhanced type hinting capabilities

## Error Handling

### Numerical Stability
- Bounds checking for population values
- NaN/Infinity detection and recovery
- Graceful degradation for edge cases

### Input Validation
- Parameter range validation
- Type checking at runtime
- Meaningful error messages

## Testing Strategy

### Unit Tests
- Mathematical correctness of Lanchester solver
- Markov chain state transitions
- Termination condition detection

### Integration Tests
- End-to-end simulation scenarios
- Known outcome verification
- Performance benchmarks

### Property-Based Testing
- Invariant checking (population conservation)
- Boundary condition testing
- Stochastic behavior validation

## Extensibility Points

### Custom Distributions
- Plugin architecture for new probability distributions
- Configurable Markov chain parameters
- User-defined coefficient functions

### Alternative Solvers
- Swappable numerical integration methods
- Adaptive time-stepping options
- Higher-order accuracy schemes

### Output Formats
- Multiple result serialization formats
- Real-time streaming interfaces
- Visualization hooks

## Security Considerations

### Input Sanitization
- Parameter bounds enforcement
- Injection attack prevention
- Resource consumption limits

### Data Privacy
- No persistent storage of simulation data
- Configurable logging levels
- Anonymization of input parameters

---

This architecture provides a solid foundation for the tactical simulation engine while maintaining flexibility for future enhancements and AR integration.