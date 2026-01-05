"""
Unit tests for the tactical analysis module.

Tests the 3-7-12 heuristic framework and tactical recommendation logic.

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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posterity.analysis.tactics import (
    TacticalBrain,
    TacticalRecommendation,
    ApproachStrategy,
    TargetStrategy,
    analyze_grocery_store_scenario
)
from posterity.core.simulation import (
    SimulationResult,
    TerminationReason,
    run_tactical_simulation
)


class TestTacticalBrain:
    """Test suite for TacticalBrain class."""
    
    def test_initialization(self):
        """Test proper initialization of TacticalBrain."""
        brain = TacticalBrain(
            volatility_threshold_low=0.05,
            volatility_threshold_high=0.25,
            dominance_threshold=1.5,
            heat_threshold_high=0.7
        )
        
        assert brain.volatility_threshold_low == 0.05
        assert brain.volatility_threshold_high == 0.25
        assert brain.dominance_threshold == 1.5
        assert brain.heat_threshold_high == 0.7
    
    def test_double_crossover_longevity_strategy(self):
        """Test that double crossover leads to longevity strategy (Target 12)."""
        brain = TacticalBrain()
        
        # Create mock result with double crossover
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 90.0, 110.0],
            [2.0, 110.0, 90.0],
            [3.0, 80.0, 120.0]
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.DOUBLE_CROSSOVER,
            termination_time=3.0,
            final_bison=80.0,
            final_cattle=120.0,
            crossover_points=[1.5, 2.5],
            alpha_coefficients=[0.5, 0.4, 0.6, 0.3],
            beta_coefficients=[0.3, 0.5, 0.4, 0.7]
        )
        
        recommendation = brain.analyze_simulation(result, original_heat=0.5, original_pace=0.5)
        
        assert recommendation.target_group == TargetStrategy.LONGEVITY
        assert "longevity" in recommendation.reasoning.lower() or "12" in recommendation.reasoning
    
    def test_annihilation_strike_strategy(self):
        """Test that quick annihilation leads to strike strategy (Target 3)."""
        brain = TacticalBrain()
        
        # Create mock result with quick annihilation
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [0.5, 80.0, 90.0],
            [1.0, 60.0, 70.0],
            [1.2, 40.0, 0.1]  # Quick annihilation
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.ANNIHILATION,
            termination_time=1.2,  # Very quick
            final_bison=40.0,
            final_cattle=0.1,
            crossover_points=[],
            alpha_coefficients=[0.5, 0.4, 0.6, 0.3],
            beta_coefficients=[0.3, 0.5, 0.4, 0.7]
        )
        
        recommendation = brain.analyze_simulation(result, original_heat=0.5, original_pace=0.5)
        
        assert recommendation.target_group == TargetStrategy.STRIKE
        assert "strike" in recommendation.reasoning.lower() or "3" in recommendation.reasoning
    
    def test_crossover_balance_strategy(self):
        """Test that single crossover leads to balance strategy (Target 7)."""
        brain = TacticalBrain()
        
        # Create mock result with single crossover
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 90.0, 110.0],
            [2.0, 80.0, 120.0],
            [3.0, 70.0, 130.0]
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.CROSSOVER,
            termination_time=3.0,
            final_bison=70.0,
            final_cattle=130.0,
            crossover_points=[1.5],
            alpha_coefficients=[0.5, 0.4, 0.6, 0.3],
            beta_coefficients=[0.3, 0.5, 0.4, 0.7]
        )
        
        recommendation = brain.analyze_simulation(result, original_heat=0.5, original_pace=0.5)
        
        assert recommendation.target_group == TargetStrategy.BALANCE
        assert "balance" in recommendation.reasoning.lower() or "7" in recommendation.reasoning
    
    def test_high_heat_active_approach(self):
        """Test that high heat leads to active approach."""
        brain = TacticalBrain(heat_threshold_high=0.6)
        
        # Create mock result
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 120.0, 80.0]  # Bison wins
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=1.0,
            final_bison=120.0,
            final_cattle=80.0,
            crossover_points=[],
            alpha_coefficients=[0.5, 0.4],
            beta_coefficients=[0.3, 0.5]
        )
        
        recommendation = brain.analyze_simulation(result, original_heat=0.8, original_pace=0.5)  # High heat
        
        assert recommendation.approach == ApproachStrategy.ACTIVE
        assert "active" in recommendation.reasoning.lower()
    
    def test_low_heat_passive_approach(self):
        """Test that low heat with cattle win leads to passive approach."""
        brain = TacticalBrain(heat_threshold_high=0.6)
        
        # Create mock result with cattle winning
        trajectory = np.array([
            [0.0, 100.0, 100.0],
            [1.0, 80.0, 120.0]  # Cattle wins
        ])
        
        result = SimulationResult(
            trajectory=trajectory,
            termination_reason=TerminationReason.TIME_LIMIT,
            termination_time=1.0,
            final_bison=80.0,
            final_cattle=120.0,
            crossover_points=[],
            alpha_coefficients=[0.5, 0.4],
            beta_coefficients=[0.3, 0.5]
        )
        
        recommendation = brain.analyze_simulation(result, original_heat=0.3, original_pace=0.3)  # Low heat
        
        assert recommendation.approach == ApproachStrategy.PASSIVE
        assert "passive" in recommendation.reasoning.lower()
    
    def test_confidence_calculation(self):
        """Test confidence calculation logic."""
        brain = TacticalBrain()
        
        # High confidence scenario: clear double crossover
        high_conf_result = SimulationResult(
            trajectory=np.array([[0.0, 100.0, 100.0], [3600.0, 80.0, 20.0]]),  # Long duration
            termination_reason=TerminationReason.DOUBLE_CROSSOVER,
            termination_time=3600.0,
            final_bison=80.0,
            final_cattle=20.0,  # Clear winner
            crossover_points=[1000.0, 2000.0],
            alpha_coefficients=[0.5, 0.4],
            beta_coefficients=[0.3, 0.5]
        )
        
        high_recommendation = brain.analyze_simulation(high_conf_result, original_heat=0.5, original_pace=0.5)
        
        # Low confidence scenario: numerical instability
        low_conf_result = SimulationResult(
            trajectory=np.array([[0.0, 100.0, 100.0], [100.0, 50.0, 50.0]]),  # Short duration
            termination_reason=TerminationReason.NUMERICAL_INSTABILITY,
            termination_time=100.0,
            final_bison=50.0,
            final_cattle=50.0,  # No clear winner
            crossover_points=[],
            alpha_coefficients=[0.5, 0.4],
            beta_coefficients=[0.3, 0.5]
        )
        
        low_recommendation = brain.analyze_simulation(low_conf_result, original_heat=0.5, original_pace=0.5)
        
        assert high_recommendation.confidence > low_recommendation.confidence
        assert high_recommendation.confidence > 0.5
        assert low_recommendation.confidence < 0.5


class TestTacticalRecommendation:
    """Test suite for TacticalRecommendation class."""
    
    def test_recommendation_string_representation(self):
        """Test string representation of recommendations."""
        active_strike = TacticalRecommendation(
            approach=ApproachStrategy.ACTIVE,
            target_group=TargetStrategy.STRIKE,
            confidence=0.8,
            reasoning="Test reasoning"
        )
        
        assert str(active_strike) == "Approach as Active, Target Group 3"
        
        passive_longevity = TacticalRecommendation(
            approach=ApproachStrategy.PASSIVE,
            target_group=TargetStrategy.LONGEVITY,
            confidence=0.6,
            reasoning="Test reasoning"
        )
        
        assert str(passive_longevity) == "Approach as Passive, Target Group 12"


class TestGroceryStoreScenario:
    """Test suite for the grocery store scenario."""
    
    def test_grocery_store_scenario(self):
        """Test the classic grocery store scenario (Low Heat, Low Pace)."""
        recommendation = analyze_grocery_store_scenario()
        
        # Should result in passive recommendation as per specification
        assert recommendation.approach == ApproachStrategy.PASSIVE
        assert isinstance(recommendation.target_group, TargetStrategy)
        assert 0.0 <= recommendation.confidence <= 1.0
        assert len(recommendation.reasoning) > 0
        
        # Verify it's a valid recommendation string
        rec_str = str(recommendation)
        assert "Passive" in rec_str
        assert "Target Group" in rec_str
    
    def test_grocery_store_reproducibility(self):
        """Test that grocery store scenario is reproducible."""
        rec1 = analyze_grocery_store_scenario()
        rec2 = analyze_grocery_store_scenario()
        
        # Should be identical due to fixed seed
        assert rec1.approach == rec2.approach
        assert rec1.target_group == rec2.target_group
        assert rec1.confidence == rec2.confidence


class TestIntegrationScenarios:
    """Integration tests with real simulations."""
    
    def test_high_volatility_integration(self):
        """Test tactical analysis with high volatility simulation."""
        # Run a high volatility simulation
        result = run_tactical_simulation(
            pace=0.7,
            flux=0.2,
            heat=0.9,  # Very high heat
            count=50.0,
            random_seed=123,
            max_hours=0.2
        )
        
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(result, original_heat=0.9, original_pace=0.7)
        
        # High heat should generally lead to active approach
        assert recommendation.approach == ApproachStrategy.ACTIVE
        assert isinstance(recommendation.target_group, TargetStrategy)
        assert recommendation.confidence > 0.0
    
    def test_low_volatility_integration(self):
        """Test tactical analysis with low volatility simulation."""
        # Run a low volatility simulation
        result = run_tactical_simulation(
            pace=0.2,
            flux=0.1,
            heat=0.1,  # Very low heat
            count=100.0,
            random_seed=456,
            max_hours=0.2
        )
        
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(result, original_heat=0.1, original_pace=0.2)
        
        # Low heat should generally lead to passive approach
        assert recommendation.approach == ApproachStrategy.PASSIVE
        assert isinstance(recommendation.target_group, TargetStrategy)
        assert recommendation.confidence > 0.0
    
    def test_balanced_scenario_integration(self):
        """Test tactical analysis with balanced parameters."""
        # Run a balanced simulation
        result = run_tactical_simulation(
            pace=0.5,
            flux=0.0,  # No initial imbalance
            heat=0.5,  # Medium heat
            count=75.0,
            random_seed=789,
            max_hours=0.3
        )
        
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(result, original_heat=0.5, original_pace=0.5)
        
        # Balanced scenario should often lead to balance strategy
        assert isinstance(recommendation.approach, ApproachStrategy)
        assert isinstance(recommendation.target_group, TargetStrategy)
        assert recommendation.confidence > 0.0
        
        # Should have reasonable reasoning
        assert len(recommendation.reasoning) > 20  # Non-trivial explanation


if __name__ == "__main__":
    pytest.main([__file__])