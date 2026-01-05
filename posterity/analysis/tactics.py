"""
Tactical heuristics implementation for the 3-7-12 decision framework.

This module interprets simulation results and provides tactical advice based on
the volatility patterns and outcomes of the Lanchester-Markov simulation.

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

from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.simulation import SimulationResult, TerminationReason


class ApproachStrategy(Enum):
    """Approach strategies for social dynamics."""
    ACTIVE = "active"
    PASSIVE = "passive"


class TargetStrategy(Enum):
    """Target group size strategies based on 3-7-12 framework."""
    STRIKE = 3      # Target group of 3 - Quick, decisive action
    BALANCE = 7     # Target group of 7 - Balanced approach
    LONGEVITY = 12  # Target group of 12 - Long-term, patient strategy


@dataclass
class TacticalRecommendation:
    """Complete tactical recommendation from simulation analysis."""
    approach: ApproachStrategy
    target_group: TargetStrategy
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    def __str__(self) -> str:
        """Human-readable recommendation string."""
        approach_str = "Active" if self.approach == ApproachStrategy.ACTIVE else "Passive"
        return f"Approach as {approach_str}, Target Group {self.target_group.value}"


class TacticalBrain:
    """
    Heuristic interpretation layer implementing the 3-7-12 tactical framework.
    
    This class analyzes simulation results and provides tactical recommendations
    based on volatility patterns, termination conditions, and population dynamics.
    """
    
    def __init__(
        self,
        volatility_threshold_low: float = 0.1,
        volatility_threshold_high: float = 0.3,
        dominance_threshold: float = 1.2,
        heat_threshold_high: float = 0.6
    ):
        """
        Initialize the tactical brain with configurable thresholds.
        
        Args:
            volatility_threshold_low: Threshold for low volatility scenarios
            volatility_threshold_high: Threshold for high volatility scenarios
            dominance_threshold: Ratio threshold for determining dominance
            heat_threshold_high: Threshold for high heat scenarios
        """
        self.volatility_threshold_low = volatility_threshold_low
        self.volatility_threshold_high = volatility_threshold_high
        self.dominance_threshold = dominance_threshold
        self.heat_threshold_high = heat_threshold_high
    
    def analyze_simulation(
        self,
        result: SimulationResult,
        original_heat: float,
        original_pace: float
    ) -> TacticalRecommendation:
        """
        Analyze simulation results and provide tactical recommendation.
        
        Args:
            result: Complete simulation results
            original_heat: Original heat parameter used in simulation
            original_pace: Original pace parameter used in simulation
            
        Returns:
            Complete tactical recommendation
        """
        # Determine target strategy based on termination pattern
        target_strategy = self._determine_target_strategy(result)
        
        # Determine approach strategy based on heat and outcome
        approach_strategy = self._determine_approach_strategy(
            result, original_heat, original_pace
        )
        
        # Calculate confidence based on simulation clarity
        confidence = self._calculate_confidence(result)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            result, target_strategy, approach_strategy, original_heat
        )
        
        return TacticalRecommendation(
            approach=approach_strategy,
            target_group=target_strategy,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _determine_target_strategy(self, result: SimulationResult) -> TargetStrategy:
        """
        Determine target group size based on simulation termination pattern.
        
        The 3-7-12 Logic:
        - Double Crossover (complex dance) → Longevity strategy (Target 12)
        - Fast dominance (quick win) → Strike strategy (Target 3)
        - Stable/balanced outcome → Balance strategy (Target 7)
        """
        if result.termination_reason == TerminationReason.DOUBLE_CROSSOVER:
            # Complex dance pattern indicates need for patient, long-term approach
            return TargetStrategy.LONGEVITY
        
        elif result.termination_reason == TerminationReason.ANNIHILATION:
            # Quick, decisive outcome suggests strike strategy is effective
            # But check if it was too quick (might need more patience)
            if result.duration < 1800:  # Less than 30 minutes
                return TargetStrategy.STRIKE
            else:
                return TargetStrategy.BALANCE
        
        elif result.termination_reason == TerminationReason.CROSSOVER:
            # Single reversal suggests balanced approach
            return TargetStrategy.BALANCE
        
        else:  # TIME_LIMIT or other
            # Stable, long-running simulation suggests balanced approach
            # unless there's high volatility
            trajectory = result.trajectory
            if len(trajectory) > 1:
                bison_volatility = np.std(trajectory[:, 1])
                cattle_volatility = np.std(trajectory[:, 2])
                avg_volatility = (bison_volatility + cattle_volatility) / 2
                
                if avg_volatility > self.volatility_threshold_high:
                    return TargetStrategy.LONGEVITY
                else:
                    return TargetStrategy.BALANCE
            
            return TargetStrategy.BALANCE
    
    def _determine_approach_strategy(
        self,
        result: SimulationResult,
        heat: float,
        pace: float
    ) -> ApproachStrategy:
        """
        Determine approach strategy based on heat, pace, and outcome.
        
        Active/Passive Logic:
        - High Heat + Bison Win → Active Approach
        - Low Heat + Cattle Win → Passive Grazing
        - High Pace + Volatility → Active Approach
        - Low Pace + Stability → Passive Approach
        """
        # Analyze winner and dominance
        bison_won = result.winner == "bison"
        cattle_won = result.winner == "cattle"
        
        # High heat scenarios
        if heat > self.heat_threshold_high:
            if bison_won or result.termination_reason == TerminationReason.DOUBLE_CROSSOVER:
                return ApproachStrategy.ACTIVE
            else:
                # High heat but cattle won - still suggests active approach
                # because high volatility requires active management
                return ApproachStrategy.ACTIVE
        
        # Low heat scenarios
        else:
            if cattle_won and result.termination_reason != TerminationReason.DOUBLE_CROSSOVER:
                return ApproachStrategy.PASSIVE
            elif bison_won and pace > 0.5:
                return ApproachStrategy.ACTIVE
            else:
                # Default to passive for low heat, stable scenarios
                return ApproachStrategy.PASSIVE
    
    def _calculate_confidence(self, result: SimulationResult) -> float:
        """
        Calculate confidence level based on simulation clarity and consistency.
        
        Higher confidence for:
        - Clear termination reasons
        - Consistent population trends
        - Sufficient simulation duration
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear termination reasons
        if result.termination_reason in [
            TerminationReason.DOUBLE_CROSSOVER,
            TerminationReason.ANNIHILATION
        ]:
            confidence += 0.3
        elif result.termination_reason == TerminationReason.CROSSOVER:
            confidence += 0.2
        
        # Boost confidence for sufficient duration
        if result.duration > 1800:  # More than 30 minutes
            confidence += 0.1
        elif result.duration < 300:  # Less than 5 minutes
            confidence -= 0.1
        
        # Boost confidence for clear winner
        if result.winner is not None:
            final_ratio = max(result.final_bison, result.final_cattle) / max(
                min(result.final_bison, result.final_cattle), 0.1
            )
            if final_ratio > self.dominance_threshold:
                confidence += 0.1
        
        # Penalize for numerical instability
        if result.termination_reason == TerminationReason.NUMERICAL_INSTABILITY:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(
        self,
        result: SimulationResult,
        target: TargetStrategy,
        approach: ApproachStrategy,
        heat: float
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        reasoning_parts = []
        
        # Explain target strategy
        if target == TargetStrategy.LONGEVITY:
            if result.termination_reason == TerminationReason.DOUBLE_CROSSOVER:
                reasoning_parts.append(
                    "Complex dance pattern with multiple crossovers indicates "
                    "need for patient, long-term strategy (Target 12)"
                )
            else:
                reasoning_parts.append(
                    "High volatility detected, suggesting longevity approach (Target 12)"
                )
        
        elif target == TargetStrategy.STRIKE:
            reasoning_parts.append(
                f"Quick resolution ({result.duration:.0f}s) suggests "
                "decisive strike strategy is effective (Target 3)"
            )
        
        else:  # BALANCE
            reasoning_parts.append(
                "Balanced dynamics suggest moderate approach (Target 7)"
            )
        
        # Explain approach strategy
        if approach == ApproachStrategy.ACTIVE:
            if heat > self.heat_threshold_high:
                reasoning_parts.append(
                    f"High heat ({heat:.2f}) requires active engagement"
                )
            elif result.winner == "bison":
                reasoning_parts.append(
                    "Bison dominance indicates active approach is favorable"
                )
            else:
                reasoning_parts.append(
                    "Volatility patterns suggest active management needed"
                )
        
        else:  # PASSIVE
            if heat <= self.heat_threshold_high:
                reasoning_parts.append(
                    f"Low heat ({heat:.2f}) supports passive grazing approach"
                )
            if result.winner == "cattle":
                reasoning_parts.append(
                    "Cattle dominance suggests passive approach is effective"
                )
        
        return ". ".join(reasoning_parts) + "."


def analyze_grocery_store_scenario(
    heat: float = 0.2,
    pace: float = 0.3,
    flux: float = 0.1,
    count: float = 50.0
) -> TacticalRecommendation:
    """
    Analyze the classic 'Grocery Store' scenario (Low Heat, Low Pace).
    
    This scenario should typically result in a 'Passive' recommendation
    as per the original specification.
    
    Args:
        heat: Volatility parameter (default low)
        pace: Speed parameter (default low)
        flux: Rate of change (default low)
        count: Population size (default moderate)
        
    Returns:
        Tactical recommendation for grocery store scenario
    """
    from ..core.simulation import run_tactical_simulation
    
    # Run simulation with grocery store parameters
    result = run_tactical_simulation(
        pace=pace,
        flux=flux,
        heat=heat,
        count=count,
        random_seed=42,  # Fixed seed for reproducible grocery store analysis
        max_hours=0.5    # 30 minutes max
    )
    
    # Analyze with tactical brain
    brain = TacticalBrain()
    recommendation = brain.analyze_simulation(result, heat, pace)
    
    return recommendation