"""
Simulation runner implementing termination conditions and trajectory analysis.

This module provides the main simulation loop that orchestrates the Lanchester
solver and implements the various termination conditions for the tactical
simulation engine.

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

from typing import List, Tuple, Optional, Literal, NamedTuple
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum

from .physics import LanchesterSolver, create_solver_from_params


class TerminationReason(Enum):
    """Possible reasons for simulation termination."""
    ANNIHILATION = "annihilation"
    CROSSOVER = "crossover"
    DOUBLE_CROSSOVER = "double_crossover"
    TIME_LIMIT = "time_limit"
    NUMERICAL_INSTABILITY = "numerical_instability"


@dataclass
class SimulationResult:
    """Results from a complete simulation run."""
    trajectory: NDArray[np.float64]  # Shape: (n_steps, 3) [time, bison, cattle]
    termination_reason: TerminationReason
    termination_time: float
    final_bison: float
    final_cattle: float
    crossover_points: List[float]  # Times when populations crossed
    alpha_coefficients: List[float]  # Alpha coefficient history
    beta_coefficients: List[float]   # Beta coefficient history
    
    @property
    def duration(self) -> float:
        """Total simulation duration."""
        return self.termination_time
    
    @property
    def num_crossovers(self) -> int:
        """Number of times populations crossed."""
        return len(self.crossover_points)
    
    @property
    def winner(self) -> Optional[Literal["bison", "cattle"]]:
        """Which population won, if any."""
        if self.final_bison > self.final_cattle:
            return "bison"
        elif self.final_cattle > self.final_bison:
            return "cattle"
        return None


class SimulationRunner:
    """
    Main simulation runner that orchestrates the complete simulation lifecycle.
    
    This class manages the simulation loop, monitors termination conditions,
    and provides comprehensive trajectory analysis.
    """
    
    def __init__(
        self,
        max_simulation_hours: float = 1.5,
        min_population_threshold: float = 0.2,
        crossover_detection_threshold: float = 0.01
    ):
        """
        Initialize the simulation runner.
        
        Args:
            max_simulation_hours: Maximum simulation time in hours
            min_population_threshold: Population threshold for annihilation
            crossover_detection_threshold: Minimum difference for crossover detection
        """
        self.max_simulation_hours = max_simulation_hours
        self.min_population_threshold = min_population_threshold
        self.crossover_detection_threshold = crossover_detection_threshold
        
        # Simulation state
        self.solver: Optional[LanchesterSolver] = None
        self.crossover_history: List[float] = []
        self.last_bison_dominant: Optional[bool] = None
    
    def run_simulation(
        self,
        pace: float,
        flux: float,
        heat: float,
        count: float,
        random_seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run a complete simulation with the given parameters.
        
        Args:
            pace: Speed of iteration (affects time step)
            flux: Rate of change in the system
            heat: Volatility of Markov chain
            count: Initial population size
            random_seed: Optional seed for reproducibility
            
        Returns:
            Complete simulation results
        """
        # Create solver from parameters
        self.solver = create_solver_from_params(
            flux=flux,
            heat=heat,
            pace=pace,
            count=count,
            random_seed=random_seed
        )
        
        # Reset state
        self.crossover_history.clear()
        self.last_bison_dominant = None
        
        # Determine initial dominance
        if self.solver.bison > self.solver.cattle:
            self.last_bison_dominant = True
        elif self.solver.cattle > self.solver.bison:
            self.last_bison_dominant = False
        
        # Convert simulation time to solver time units
        max_time = self.max_simulation_hours * 3600.0  # Convert hours to seconds
        
        # Run simulation loop
        termination_reason = self._run_simulation_loop(max_time)
        
        # Get final trajectory and coefficient history
        trajectory = self.solver.get_trajectory()
        alpha_coeffs, beta_coeffs = self.solver.get_coefficient_history()
        
        # Create result
        return SimulationResult(
            trajectory=trajectory,
            termination_reason=termination_reason,
            termination_time=self.solver.time,
            final_bison=self.solver.bison,
            final_cattle=self.solver.cattle,
            crossover_points=self.crossover_history.copy(),
            alpha_coefficients=alpha_coeffs,
            beta_coefficients=beta_coeffs
        )
    
    def _run_simulation_loop(self, max_time: float) -> TerminationReason:
        """
        Execute the main simulation loop until termination condition is met.
        
        Args:
            max_time: Maximum simulation time
            
        Returns:
            Reason for termination
        """
        while self.solver.time < max_time:
            try:
                # Perform one simulation step
                bison, cattle = self.solver.step()
                
                # Check for annihilation
                if bison <= self.min_population_threshold or cattle <= self.min_population_threshold:
                    return TerminationReason.ANNIHILATION
                
                # Check for crossover
                crossover_detected = self._check_crossover(bison, cattle)
                if crossover_detected:
                    # Check for double crossover
                    if len(self.crossover_history) >= 2:
                        return TerminationReason.DOUBLE_CROSSOVER
                    # Single crossover - continue simulation to see if there's a second
                
                # Check for numerical instability
                if (np.isnan(bison) or np.isnan(cattle) or 
                    np.isinf(bison) or np.isinf(cattle)):
                    return TerminationReason.NUMERICAL_INSTABILITY
                
            except Exception as e:
                # Handle any unexpected errors
                return TerminationReason.NUMERICAL_INSTABILITY
        
        # If we've reached here, we hit the time limit
        # Determine final termination reason based on crossover history
        if len(self.crossover_history) >= 2:
            return TerminationReason.DOUBLE_CROSSOVER
        elif len(self.crossover_history) == 1:
            return TerminationReason.CROSSOVER
        else:
            return TerminationReason.TIME_LIMIT
    
    def _check_crossover(self, bison: float, cattle: float) -> bool:
        """
        Check if a population crossover has occurred.
        
        Args:
            bison: Current bison population
            cattle: Current cattle population
            
        Returns:
            True if crossover detected
        """
        # Determine current dominance
        if abs(bison - cattle) < self.crossover_detection_threshold:
            # Populations are too close to determine dominance
            return False
        
        current_bison_dominant = bison > cattle
        
        # Check if dominance has changed
        if (self.last_bison_dominant is not None and 
            self.last_bison_dominant != current_bison_dominant):
            
            # Record crossover (use current time if solver available, otherwise 0)
            current_time = self.solver.time if self.solver else 0.0
            self.crossover_history.append(current_time)
            self.last_bison_dominant = current_bison_dominant
            return True
        
        # Update dominance state
        if self.last_bison_dominant is None:
            self.last_bison_dominant = current_bison_dominant
        
        return False
    
    def analyze_trajectory(self, result: SimulationResult) -> dict:
        """
        Perform detailed analysis of simulation trajectory.
        
        Args:
            result: Simulation result to analyze
            
        Returns:
            Dictionary containing analysis metrics
        """
        trajectory = result.trajectory
        times = trajectory[:, 0]
        bison_pop = trajectory[:, 1]
        cattle_pop = trajectory[:, 2]
        
        # Calculate basic statistics
        analysis = {
            'duration': result.duration,
            'num_steps': len(trajectory),
            'termination_reason': result.termination_reason.value,
            'num_crossovers': result.num_crossovers,
            'winner': result.winner,
            
            # Population statistics
            'initial_bison': bison_pop[0],
            'initial_cattle': cattle_pop[0],
            'final_bison': result.final_bison,
            'final_cattle': result.final_cattle,
            'max_bison': np.max(bison_pop),
            'max_cattle': np.max(cattle_pop),
            'min_bison': np.min(bison_pop),
            'min_cattle': np.min(cattle_pop),
            
            # Volatility measures
            'bison_volatility': np.std(bison_pop),
            'cattle_volatility': np.std(cattle_pop),
            'total_volatility': np.std(bison_pop + cattle_pop),
            
            # Crossover analysis
            'crossover_times': result.crossover_points,
            'time_to_first_crossover': result.crossover_points[0] if result.crossover_points else None,
        }
        
        # Calculate attrition rates
        if len(trajectory) > 1:
            bison_attrition = (bison_pop[0] - result.final_bison) / result.duration
            cattle_attrition = (cattle_pop[0] - result.final_cattle) / result.duration
            analysis['bison_attrition_rate'] = bison_attrition
            analysis['cattle_attrition_rate'] = cattle_attrition
        
        # Determine simulation complexity
        if result.termination_reason == TerminationReason.DOUBLE_CROSSOVER:
            analysis['complexity'] = 'high'  # Complex dance
        elif result.termination_reason == TerminationReason.CROSSOVER:
            analysis['complexity'] = 'medium'  # Single reversal
        elif result.termination_reason == TerminationReason.ANNIHILATION:
            analysis['complexity'] = 'low'  # Decisive outcome
        else:
            analysis['complexity'] = 'medium'  # Stable or time-limited
        
        return analysis


def run_tactical_simulation(
    pace: float,
    flux: float,
    heat: float,
    count: float,
    random_seed: Optional[int] = None,
    max_hours: float = 1.5
) -> SimulationResult:
    """
    Convenience function to run a single tactical simulation.
    
    Args:
        pace: Speed of iteration
        flux: Rate of change
        heat: Volatility
        count: Initial population
        random_seed: Optional seed for reproducibility
        max_hours: Maximum simulation time in hours
        
    Returns:
        Complete simulation results
    """
    runner = SimulationRunner(max_simulation_hours=max_hours)
    return runner.run_simulation(
        pace=pace,
        flux=flux,
        heat=heat,
        count=count,
        random_seed=random_seed
    )