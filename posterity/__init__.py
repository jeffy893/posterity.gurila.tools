"""
Posterity.gurila.tools - Tactical Simulation Engine

A Python 3.10+ implementation of tactical simulation using Lanchester Laws
and Markov Chain dynamics for modeling social dynamics.

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
"""

__version__ = "1.0.0"
__author__ = "Jefferson Richards <jefferson@richards.plus>"
__license__ = "GPL-3.0-or-later"

from .core import MarkovChain, LanchesterSolver, SimulationRunner, run_tactical_simulation
from .analysis import TacticalBrain, TacticalRecommendation, ApproachStrategy, TargetStrategy
from .interfaces import SceneAnalyzer, ARIntegrationHooks, CrowdAnalysisResult

__all__ = [
    'MarkovChain', 'LanchesterSolver', 'SimulationRunner', 'run_tactical_simulation',
    'TacticalBrain', 'TacticalRecommendation', 'ApproachStrategy', 'TargetStrategy',
    'SceneAnalyzer', 'ARIntegrationHooks', 'CrowdAnalysisResult'
]