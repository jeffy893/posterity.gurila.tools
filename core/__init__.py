"""
Core mathematical engine for posterity.gurila.tools

This module contains the fundamental mathematical components:
- Lanchester equation solvers
- Markov chain implementations
- Stochastic coefficient generation
"""

from .physics import MarkovChain, LanchesterSolver

__all__ = ['MarkovChain', 'LanchesterSolver']