"""
Core mathematical engine implementing Lanchester Laws and Markov Chain dynamics.

This module provides the fundamental mathematical components for the tactical
simulation engine, including stochastic coefficient generation and numeric
solving of differential equations.
"""

from typing import List, Tuple, Optional, Literal
import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
from dataclasses import dataclass
from enum import Enum


class DistributionType(Enum):
    """Supported probability distributions for Markov chain transitions."""
    NORMAL = "normal"
    LEFT_SKEWED = "left_skewed"
    RIGHT_SKEWED = "right_skewed"


@dataclass
class MoraleState:
    """Represents a state in the Markov chain with associated coefficient."""
    state_id: int
    coefficient: float
    transition_probs: NDArray[np.float64]


class MarkovChain:
    """
    Markov Chain implementation for generating stochastic coefficients.
    
    This class manages the current state of "Morale" and provides transitions
    between states. Each state maps to a floating-point coefficient used in
    the Lanchester equations.
    """
    
    def __init__(
        self, 
        num_states: int = 10,
        heat: float = 0.5,
        distribution_type: DistributionType = DistributionType.NORMAL,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Markov Chain.
        
        Args:
            num_states: Number of discrete states in the chain
            heat: Volatility parameter (0.0 to 1.0)
            distribution_type: Type of distribution for coefficient generation
            random_seed: Optional seed for reproducible results
        """
        if not 0.0 <= heat <= 1.0:
            raise ValueError("Heat must be between 0.0 and 1.0")
        
        self.num_states = num_states
        self.heat = heat
        self.distribution_type = distribution_type
        self.current_state = num_states // 2  # Start in middle state
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate transition matrix and coefficients
        self._generate_transition_matrix()
        self._generate_coefficients()
    
    def _generate_transition_matrix(self) -> None:
        """Generate the transition probability matrix based on heat parameter."""
        # Higher heat = more volatile transitions
        volatility = self.heat * 2.0  # Scale heat to reasonable range
        
        # Create transition matrix with bias toward staying in current state
        # but with heat-dependent probability of jumping
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        
        for i in range(self.num_states):
            # Base probability of staying in current state
            stay_prob = 0.7 - (volatility * 0.3)
            
            # Probability of moving to adjacent states
            move_prob = (1.0 - stay_prob) / 2.0
            
            # Set probabilities
            self.transition_matrix[i, i] = stay_prob
            
            if i > 0:
                self.transition_matrix[i, i-1] = move_prob
            if i < self.num_states - 1:
                self.transition_matrix[i, i+1] = move_prob
            
            # Normalize to ensure probabilities sum to 1
            self.transition_matrix[i] /= np.sum(self.transition_matrix[i])
    
    def _generate_coefficients(self) -> None:
        """Generate coefficients for each state based on distribution type."""
        if self.distribution_type == DistributionType.NORMAL:
            self.coefficients = stats.norm.rvs(
                loc=0.5, scale=0.2, size=self.num_states
            )
        elif self.distribution_type == DistributionType.LEFT_SKEWED:
            # Use beta distribution with alpha < beta for left skew
            self.coefficients = stats.beta.rvs(
                a=2, b=5, size=self.num_states
            )
        elif self.distribution_type == DistributionType.RIGHT_SKEWED:
            # Use beta distribution with alpha > beta for right skew
            self.coefficients = stats.beta.rvs(
                a=5, b=2, size=self.num_states
            )
        
        # Ensure coefficients are positive and reasonable
        self.coefficients = np.abs(self.coefficients)
        self.coefficients = np.clip(self.coefficients, 0.01, 2.0)
    
    def transition(self) -> float:
        """
        Perform one transition step and return the new coefficient.
        
        Returns:
            The coefficient associated with the new state
        """
        # Sample next state based on transition probabilities
        next_state = np.random.choice(
            self.num_states, 
            p=self.transition_matrix[self.current_state]
        )
        
        self.current_state = next_state
        return self.coefficients[self.current_state]
    
    def get_current_coefficient(self) -> float:
        """Get the coefficient for the current state."""
        return self.coefficients[self.current_state]
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_state = self.num_states // 2


class LanchesterSolver:
    """
    Numeric solver for Lanchester Square Law equations with dynamic coefficients.
    
    Implements the differential equations:
    d(Bison)/dt = -beta * Cattle
    d(Cattle)/dt = -alpha * Bison
    
    Where alpha and beta are updated by Markov Chain at each time step.
    """
    
    def __init__(
        self,
        initial_bison: float,
        initial_cattle: float,
        markov_alpha: MarkovChain,
        markov_beta: MarkovChain,
        dt: float = 0.01
    ):
        """
        Initialize the Lanchester solver.
        
        Args:
            initial_bison: Initial Bison population
            initial_cattle: Initial Cattle population
            markov_alpha: Markov chain for alpha coefficient
            markov_beta: Markov chain for beta coefficient
            dt: Time step for numerical integration
        """
        if initial_bison <= 0 or initial_cattle <= 0:
            raise ValueError("Initial populations must be positive")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        
        self.initial_bison = initial_bison
        self.initial_cattle = initial_cattle
        self.markov_alpha = markov_alpha
        self.markov_beta = markov_beta
        self.dt = dt
        
        # Current state
        self.bison = initial_bison
        self.cattle = initial_cattle
        self.time = 0.0
        
        # History storage
        self.history: List[Tuple[float, float, float]] = []
        self._record_state()
    
    def _record_state(self) -> None:
        """Record current state in history."""
        self.history.append((self.time, self.bison, self.cattle))
    
    def step(self) -> Tuple[float, float]:
        """
        Perform one integration step using Euler method.
        
        Returns:
            Tuple of (bison_population, cattle_population)
        """
        # Get current coefficients from Markov chains
        alpha = self.markov_alpha.transition()
        beta = self.markov_beta.transition()
        
        # Compute derivatives
        dbison_dt = -beta * self.cattle
        dcattle_dt = -alpha * self.bison
        
        # Euler integration
        self.bison += dbison_dt * self.dt
        self.cattle += dcattle_dt * self.dt
        
        # Ensure populations don't go negative
        self.bison = max(0.0, self.bison)
        self.cattle = max(0.0, self.cattle)
        
        # Update time
        self.time += self.dt
        
        # Record state
        self._record_state()
        
        return self.bison, self.cattle
    
    def solve_until(
        self, 
        max_time: float,
        min_population: float = 0.1
    ) -> NDArray[np.float64]:
        """
        Solve the system until termination condition is met.
        
        Args:
            max_time: Maximum simulation time
            min_population: Minimum population before considering annihilation
            
        Returns:
            Array of shape (n_steps, 3) containing [time, bison, cattle]
        """
        while self.time < max_time:
            bison, cattle = self.step()
            
            # Check for annihilation
            if bison <= min_population or cattle <= min_population:
                break
            
            # Check for numerical instability
            if np.isnan(bison) or np.isnan(cattle) or np.isinf(bison) or np.isinf(cattle):
                raise RuntimeError("Numerical instability detected")
        
        return np.array(self.history)
    
    def reset(self) -> None:
        """Reset solver to initial conditions."""
        self.bison = self.initial_bison
        self.cattle = self.initial_cattle
        self.time = 0.0
        self.history.clear()
        self.markov_alpha.reset()
        self.markov_beta.reset()
        self._record_state()
    
    def get_trajectory(self) -> NDArray[np.float64]:
        """Get the complete trajectory as a numpy array."""
        return np.array(self.history)


def create_solver_from_params(
    flux: float,
    heat: float,
    pace: float,
    count: float,
    random_seed: Optional[int] = None
) -> LanchesterSolver:
    """
    Factory function to create a LanchesterSolver from simulation parameters.
    
    Args:
        flux: Rate of change parameter
        heat: Volatility parameter
        pace: Speed of iteration
        count: Initial population size
        random_seed: Optional seed for reproducibility
        
    Returns:
        Configured LanchesterSolver instance
    """
    # Convert parameters to solver configuration
    dt = pace / 100.0  # Smaller time steps for higher pace
    
    # Create Markov chains with different characteristics
    markov_alpha = MarkovChain(
        num_states=10,
        heat=heat,
        distribution_type=DistributionType.NORMAL,
        random_seed=random_seed
    )
    
    markov_beta = MarkovChain(
        num_states=10,
        heat=heat * 0.8,  # Slightly different volatility
        distribution_type=DistributionType.RIGHT_SKEWED,
        random_seed=random_seed + 1 if random_seed else None
    )
    
    # Scale initial populations based on flux
    initial_bison = count * (1.0 + flux * 0.1)
    initial_cattle = count * (1.0 - flux * 0.1)
    
    return LanchesterSolver(
        initial_bison=initial_bison,
        initial_cattle=initial_cattle,
        markov_alpha=markov_alpha,
        markov_beta=markov_beta,
        dt=dt
    )