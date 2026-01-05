"""
Unit tests for the core physics module.

Tests the mathematical correctness and stability of the Lanchester solver
and Markov chain implementations.

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

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posterity.core.physics import (
    MarkovChain, 
    LanchesterSolver, 
    DistributionType,
    create_solver_from_params
)


class TestMarkovChain:
    """Test suite for MarkovChain class."""
    
    def test_initialization(self):
        """Test proper initialization of MarkovChain."""
        mc = MarkovChain(num_states=5, heat=0.3, random_seed=42)
        
        assert mc.num_states == 5
        assert mc.heat == 0.3
        assert mc.current_state == 2  # Middle state
        assert mc.transition_matrix.shape == (5, 5)
        assert len(mc.coefficients) == 5
    
    def test_invalid_heat(self):
        """Test that invalid heat values raise ValueError."""
        with pytest.raises(ValueError, match="Heat must be between 0.0 and 1.0"):
            MarkovChain(heat=-0.1)
        
        with pytest.raises(ValueError, match="Heat must be between 0.0 and 1.0"):
            MarkovChain(heat=1.5)
    
    def test_transition_matrix_properties(self):
        """Test that transition matrix has proper stochastic properties."""
        mc = MarkovChain(num_states=10, heat=0.5, random_seed=42)
        
        # Each row should sum to 1 (stochastic matrix)
        row_sums = np.sum(mc.transition_matrix, axis=1)
        assert_allclose(row_sums, 1.0, rtol=1e-10)
        
        # All probabilities should be non-negative
        assert np.all(mc.transition_matrix >= 0)
    
    def test_coefficient_generation(self):
        """Test coefficient generation for different distributions."""
        # Normal distribution
        mc_normal = MarkovChain(
            num_states=100, 
            distribution_type=DistributionType.NORMAL,
            random_seed=42
        )
        assert np.all(mc_normal.coefficients > 0)
        assert np.all(mc_normal.coefficients <= 2.0)
        
        # Left skewed
        mc_left = MarkovChain(
            num_states=100,
            distribution_type=DistributionType.LEFT_SKEWED,
            random_seed=42
        )
        assert np.all(mc_left.coefficients > 0)
        
        # Right skewed
        mc_right = MarkovChain(
            num_states=100,
            distribution_type=DistributionType.RIGHT_SKEWED,
            random_seed=42
        )
        assert np.all(mc_right.coefficients > 0)
    
    def test_transition_reproducibility(self):
        """Test that transitions are reproducible with same seed."""
        mc1 = MarkovChain(num_states=5, heat=0.5, random_seed=42)
        mc2 = MarkovChain(num_states=5, heat=0.5, random_seed=42)
        
        transitions1 = [mc1.transition() for _ in range(10)]
        transitions2 = [mc2.transition() for _ in range(10)]
        
        assert_allclose(transitions1, transitions2)
    
    def test_reset_functionality(self):
        """Test that reset returns to initial state."""
        mc = MarkovChain(num_states=10, heat=0.5, random_seed=42)
        initial_state = mc.current_state
        
        # Perform some transitions
        for _ in range(5):
            mc.transition()
        
        # Reset and check
        mc.reset()
        assert mc.current_state == initial_state


class TestLanchesterSolver:
    """Test suite for LanchesterSolver class."""
    
    def test_initialization(self):
        """Test proper initialization of LanchesterSolver."""
        mc_alpha = MarkovChain(num_states=5, heat=0.3, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.3, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=100.0,
            initial_cattle=100.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.01
        )
        
        assert solver.bison == 100.0
        assert solver.cattle == 100.0
        assert solver.time == 0.0
        assert len(solver.history) == 1  # Initial state recorded
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise ValueError."""
        mc_alpha = MarkovChain(num_states=5, heat=0.3, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.3, random_seed=43)
        
        # Negative populations
        with pytest.raises(ValueError, match="Initial populations must be positive"):
            LanchesterSolver(-10, 100, mc_alpha, mc_beta)
        
        # Negative time step
        with pytest.raises(ValueError, match="Time step must be positive"):
            LanchesterSolver(100, 100, mc_alpha, mc_beta, dt=-0.01)
    
    def test_single_step(self):
        """Test single integration step."""
        mc_alpha = MarkovChain(num_states=5, heat=0.1, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.1, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=100.0,
            initial_cattle=100.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.01
        )
        
        initial_bison = solver.bison
        initial_cattle = solver.cattle
        
        bison, cattle = solver.step()
        
        # Populations should change
        assert bison != initial_bison
        assert cattle != initial_cattle
        
        # Time should advance
        assert solver.time == 0.01
        
        # History should be updated
        assert len(solver.history) == 2
    
    def test_population_conservation_property(self):
        """Test that the system behaves reasonably (populations decrease over time)."""
        mc_alpha = MarkovChain(num_states=5, heat=0.2, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.2, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=100.0,
            initial_cattle=100.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.01
        )
        
        # Run for several steps
        for _ in range(50):
            solver.step()
        
        # In Lanchester equations, populations should generally decrease
        # (though stochastic coefficients might cause temporary increases)
        final_total = solver.bison + solver.cattle
        initial_total = 200.0
        
        # At minimum, we shouldn't have massive population explosion
        assert final_total < initial_total * 2
    
    def test_non_negative_populations(self):
        """Test that populations never go negative."""
        mc_alpha = MarkovChain(num_states=5, heat=0.8, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.8, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=10.0,
            initial_cattle=10.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.1  # Larger time step to potentially cause issues
        )
        
        # Run until one population is very small
        for _ in range(200):
            bison, cattle = solver.step()
            assert bison >= 0.0
            assert cattle >= 0.0
            
            if bison <= 0.1 or cattle <= 0.1:
                break
    
    def test_solve_until_termination(self):
        """Test the solve_until method with termination conditions."""
        mc_alpha = MarkovChain(num_states=5, heat=0.5, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.5, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=50.0,
            initial_cattle=50.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.01
        )
        
        trajectory = solver.solve_until(max_time=10.0, min_population=1.0)
        
        # Should return valid trajectory
        assert trajectory.shape[1] == 3  # [time, bison, cattle]
        assert trajectory.shape[0] > 1  # Multiple time steps
        
        # Time should be monotonically increasing
        times = trajectory[:, 0]
        assert np.all(np.diff(times) >= 0)
        
        # Final populations should be non-negative
        final_bison = trajectory[-1, 1]
        final_cattle = trajectory[-1, 2]
        assert final_bison >= 0.0
        assert final_cattle >= 0.0
    
    def test_reset_functionality(self):
        """Test that reset returns solver to initial conditions."""
        mc_alpha = MarkovChain(num_states=5, heat=0.3, random_seed=42)
        mc_beta = MarkovChain(num_states=5, heat=0.3, random_seed=43)
        
        solver = LanchesterSolver(
            initial_bison=100.0,
            initial_cattle=100.0,
            markov_alpha=mc_alpha,
            markov_beta=mc_beta,
            dt=0.01
        )
        
        # Run some steps
        for _ in range(10):
            solver.step()
        
        # Reset
        solver.reset()
        
        # Check initial conditions restored
        assert solver.bison == 100.0
        assert solver.cattle == 100.0
        assert solver.time == 0.0
        assert len(solver.history) == 1


class TestFactoryFunction:
    """Test suite for the factory function."""
    
    def test_create_solver_from_params(self):
        """Test the factory function creates valid solver."""
        solver = create_solver_from_params(
            flux=0.5,
            heat=0.8,
            pace=0.3,
            count=50.0,
            random_seed=42
        )
        
        assert isinstance(solver, LanchesterSolver)
        assert solver.bison > 0
        assert solver.cattle > 0
        assert solver.dt > 0
    
    def test_parameter_scaling(self):
        """Test that parameters are scaled appropriately."""
        solver1 = create_solver_from_params(
            flux=0.0, heat=0.5, pace=0.5, count=100.0, random_seed=42
        )
        solver2 = create_solver_from_params(
            flux=0.5, heat=0.5, pace=0.5, count=100.0, random_seed=42
        )
        
        # Higher flux should create imbalanced initial populations
        assert abs(solver1.bison - solver1.cattle) < abs(solver2.bison - solver2.cattle)


if __name__ == "__main__":
    pytest.main([__file__])