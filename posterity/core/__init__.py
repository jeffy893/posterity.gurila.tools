"""
Core mathematical engine for posterity.gurila.tools

This module contains the fundamental mathematical components:
- Lanchester equation solvers
- Markov chain implementations
- Stochastic coefficient generation

Copyright (C) 2026 Jefferson Richards <jefferson@richards.plus>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Requires Python 3.10 or later.
"""

from .physics import MarkovChain, LanchesterSolver
from .simulation import SimulationRunner, SimulationResult, TerminationReason, run_tactical_simulation

__all__ = ['MarkovChain', 'LanchesterSolver', 'SimulationRunner', 'SimulationResult', 'TerminationReason', 'run_tactical_simulation']