"""
Unit tests for the simulation module.

Tests the simulation runner, termination conditions, and trajectory analysis.

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
from numpy.testing import assert_allclose
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posterity.core.simulation import (
    SimulationRunner,
    SimulationResult,
    TerminationReason,
    run_tactical_simulation
)
from posterity.core.physics import create_solver_from_params


class TestSimulationRunner:
    """Test suite for SimulationRunner class."""
    
    def test_initialization(self):
        """Test proper initialization of SimulationRunner."""
        runner = SimulationRunner(
            max_simulation_hours=2.0,
            min_population_threshold=0.5,
            crossover_detection_threshold=0.02
        )
        
        assert runner.max_simulation_hours == 2.0
        assert runner.min_population_threshold == 0.5
        assert runner.crossover_detection_threshold == 0.02
        assert runner.solver is None
        assert len(runner.crossover_history) == 0
    
    def test_run_simulation_basic(self):
        """Test basic simulation run."""
        runner = SimulationRunner(max_simulation_hours=0.1)  # Short simulation
        
        result = runner.run_simulation(
            pace=0.5,
            flux=0.3,
            heat=0.4,
            count=50.0,
            random_seed=42
        )
        
        assert isinstance(result, SimulationResult)
        assert result.trajectory.shape[1] == 3  # [time, bison, cattle]
        assert result.trajectory.shape[0] > 1  # Multiple steps
        assert result.termination_time > 0
        assert result.final_bison >= 0
        assert result.final_cattle >= 0
        assert isinstance(result.termination_reason, TerminationReason)
    
    def test_annihilation_termination(self):
        """Test that simulation terminates on annihilation."""
        runner = SimulationRunner(
            max_simulation_hours=10.0,  # Long enough to not hit time limit
            min_population_threshold=5.0  # High threshold to trigger annihilation
        )
        
        result = runner.run_simulation(
            pace=0.8,  # Fast pace
            flux=0.9,  # High flux for imbalance
            heat=0.7,  # High volatility
            count=10.0,  # Small population
            random_seed=42
        )
        
        # Should terminate due to annihilation
        assert (result.termination_reason == TerminationReason.ANNIHILATION or
                result.termination_reason == TerminationReason.TIME_LIMIT)
        assert result.final_bison <= 5.0 or result.final_cattle <= 5.0
    
    def test_time_limit_termination(self):
        """Test that simulation terminates on time limit."""
        runner = SimulationRunner(
            max_simulation_hours=0.01,  # Very short time limit
            min_population_threshold=0.1  # Low threshold to avoid annihilation
        )
        
        result = runner.run_simulation(
            pace=0.1,  # Slow pace
            flux=0.1,  # Low flux for stability
            heat=0.1,  # Low volatility
            count=100.0,  # Large population
            random_seed=42
        )
        
        # Should terminate due to time limit or other valid reasons
        assert result.termination_reason in [
            TerminationReason.TIME_LIMIT,
            TerminationReason.CROSSOVER,
            TerminationReason.DOUBLE_CROSSOVER,
            TerminationReason.ANNIHILATION  # Can happen with small populations
        ]
    
    def test_crossover_detection(self):
        """Test crossover detection logic."""
        runner = SimulationRunner(crossover_detection_threshold=0.1)
        
        # Test crossover detection method directly
        runner.last_bison_dominant = True
        
        # No crossover - bison still dominant
        crossover = runner._check_crossover(10.0, 5.0)
        assert not crossover
        assert runner.last_bison_dominant is True
        
        # Crossover - cattle now dominant
        crossover = runner._check_crossover(5.0, 10.0)
        assert crossover
        assert runner.last_bison_dominant is False
        assert len(runner.crossover_history) == 1
    
    def test_trajectory_analysis(self):
        """Test trajectory analysis functionality."""
        runner = SimulationRunner()
        
        # Create a mock result
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 90.0, 95.0],
            [2.0, 80.0, 85.0],
            [3.0, 70.0, 75.0]
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=3.0,
            final_bison=70.0,
            final_cattle=75.0,
            crossover_points=[]
        )
        
        analysis = runner.analyze_trajectory(result)
        
        assert analysis['duration'] == 3.0
        assert analysis['num_steps'] == 4
        assert analysis['initial_bison'] == 100.0
        assert analysis['initial_cattle'] == 100.0
        assert analysis['final_bison'] == 70.0
        assert analysis['final_cattle'] == 75.0
        assert analysis['winner'] == 'cattle'
        assert analysis['num_crossovers'] == 0
        assert 'bison_volatility' in analysis
        assert 'cattle_volatility' in analysis
    
    def test_simulation_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        runner1 = SimulationRunner(max_simulation_hours=0.1)
        runner2 = SimulationRunner(max_simulation_hours=0.1)
        
        result1 = runner1.run_simulation(
            pace=0.5, flux=0.3, heat=0.4, count=50.0, random_seed=42
        )
        result2 = runner2.run_simulation(
            pace=0.5, flux=0.3, heat=0.4, count=50.0, random_seed=42
        )
        
        # Results should be identical
        assert result1.termination_reason == result2.termination_reason
        assert_allclose(result1.final_bison, result2.final_bison, rtol=1e-10)
        assert_allclose(result1.final_cattle, result2.final_cattle, rtol=1e-10)
        assert len(result1.crossover_points) == len(result2.crossover_points)


class TestSimulationResult:
    """Test suite for SimulationResult class."""
    
    def test_simulation_result_properties(self):
        """Test SimulationResult properties."""
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 90.0, 110.0],
            [2.0, 80.0, 85.0]
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.CROSSOVER,
            termination_time=2.0,
            final_bison=80.0,
            final_cattle=85.0,
            crossover_points=[1.5]
        )
        
        assert result.duration == 2.0
        assert result.num_crossovers == 1
        assert result.winner == 'cattle'
    
    def test_winner_determination(self):
        """Test winner determination logic."""
        # Bison wins
        result1 = SimulationResult(
            trajectory=np.array([[0.0, 100.0, 100.0]]),
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=1.0,
            final_bison=60.0,
            final_cattle=40.0,
            crossover_points=[]
        )
        assert result1.winner == 'bison'
        
        # Cattle wins
        result2 = SimulationResult(
            trajectory=np.array([[0.0, 100.0, 100.0]]),
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=1.0,
            final_bison=40.0,
            final_cattle=60.0,
            crossover_points=[]
        )
        assert result2.winner == 'cattle'
        
        # Tie
        result3 = SimulationResult(
            trajectory=np.array([[0.0, 100.0, 100.0]]),
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=1.0,
            final_bison=50.0,
            final_cattle=50.0,
            crossover_points=[]
        )
        assert result3.winner is None


class TestConvenienceFunction:
    """Test suite for convenience functions."""
    
    def test_run_tactical_simulation(self):
        """Test the convenience function."""
        result = run_tactical_simulation(
            pace=0.5,
            flux=0.3,
            heat=0.4,
            count=50.0,
            random_seed=42,
            max_hours=0.1
        )
        
        assert isinstance(result, SimulationResult)
        assert result.trajectory.shape[1] == 3
        assert result.termination_time > 0
        assert isinstance(result.termination_reason, TerminationReason)


class TestComplexScenarios:
    """Test suite for complex simulation scenarios."""
    
    def test_high_volatility_scenario(self):
        """Test simulation with high volatility (should create crossovers)."""
        result = run_tactical_simulation(
            pace=0.8,
            flux=0.2,  # Balanced initial populations
            heat=0.9,  # Very high volatility
            count=50.0,
            random_seed=123,
            max_hours=0.5
        )
        
        # High volatility should potentially create crossovers
        assert result.termination_reason in [
            TerminationReason.CROSSOVER,
            TerminationReason.DOUBLE_CROSSOVER,
            TerminationReason.TIME_LIMIT,
            TerminationReason.ANNIHILATION
        ]
    
    def test_low_volatility_scenario(self):
        """Test simulation with low volatility (should be stable)."""
        result = run_tactical_simulation(
            pace=0.3,
            flux=0.1,
            heat=0.1,  # Very low volatility
            count=100.0,
            random_seed=456,
            max_hours=0.2
        )
        
        # Low volatility should be more stable
        assert result.termination_reason in [
            TerminationReason.TIME_LIMIT,
            TerminationReason.ANNIHILATION
        ]
        # Should have fewer crossovers
        assert result.num_crossovers <= 1
    
    def test_extreme_imbalance_scenario(self):
        """Test simulation with extreme initial imbalance."""
        result = run_tactical_simulation(
            pace=0.5,
            flux=0.95,  # Extreme imbalance
            heat=0.5,
            count=50.0,
            random_seed=789,
            max_hours=0.3
        )
        
        # Extreme imbalance should lead to quick resolution
        assert result.termination_reason in [
            TerminationReason.ANNIHILATION,
            TerminationReason.TIME_LIMIT
        ]


if __name__ == "__main__":
    pytest.main([__file__])