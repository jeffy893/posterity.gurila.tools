"""
Venue-specific presets for tactical simulations.

This module provides realistic parameter combinations for different social venues,
calibrated to generate interesting dynamics and complex scenarios.

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

from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
import random
import numpy as np


@dataclass
class VenuePreset:
    """
    Represents a venue-specific parameter preset for tactical simulations.
    
    Parameters are calibrated based on real-world social dynamics observed
    in different types of venues and gatherings.
    """
    name: str
    description: str
    flux_range: tuple[float, float]  # (min, max) for rate of change
    heat_range: tuple[float, float]  # (min, max) for volatility
    pace_range: tuple[float, float]  # (min, max) for interaction speed
    count_range: tuple[int, int]     # (min, max) for population size
    max_hours: float                 # Typical duration for this venue
    complexity_bias: float           # 0.0-1.0, higher = more likely to generate complex scenarios
    
    def generate_parameters(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Generate randomized parameters within the venue's ranges.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary of simulation parameters
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Apply complexity bias to parameter selection
        # Higher complexity bias pushes parameters toward ranges that create more crossovers
        bias_factor = self.complexity_bias
        
        # For flux: lower values (closer to 0) tend to create more balanced scenarios
        flux_min, flux_max = self.flux_range
        if bias_factor > 0.5:
            # Bias toward lower flux for more balanced initial conditions
            flux = np.random.beta(2, 5) * (flux_max - flux_min) + flux_min
        else:
            flux = random.uniform(flux_min, flux_max)
        
        # For heat: higher values create more volatility and potential crossovers
        heat_min, heat_max = self.heat_range
        if bias_factor > 0.5:
            # Bias toward higher heat for more volatility
            heat = np.random.beta(2, 3) * (heat_max - heat_min) + heat_min
            heat = min(heat_max, heat + bias_factor * 0.2)  # Boost heat slightly
        else:
            heat = random.uniform(heat_min, heat_max)
        
        # For pace: moderate values tend to allow complex dynamics to develop
        pace_min, pace_max = self.pace_range
        if bias_factor > 0.5:
            # Bias toward moderate pace values
            pace = np.random.beta(3, 3) * (pace_max - pace_min) + pace_min
        else:
            pace = random.uniform(pace_min, pace_max)
        
        # Count: larger populations can sustain longer dynamics
        count_min, count_max = self.count_range
        count = random.randint(count_min, count_max)
        
        # Ensure values are within valid ranges
        flux = max(0.0, min(1.0, flux))
        heat = max(0.0, min(1.0, heat))
        pace = max(0.0, min(1.0, pace))
        
        return {
            'flux': flux,
            'heat': heat,
            'pace': pace,
            'count': float(count),
            'max_hours': self.max_hours
        }


# Venue preset definitions based on real-world social dynamics
VENUE_PRESETS = {
    'cafe': VenuePreset(
        name='Cafe',
        description='Quiet coffee shop with steady, low-energy interactions',
        flux_range=(0.05, 0.15),      # Low change rate - people settle in
        heat_range=(0.2, 0.4),        # Low volatility - calm environment
        pace_range=(0.2, 0.4),        # Slow pace - relaxed interactions
        count_range=(15, 40),          # Small to medium groups
        max_hours=2.0,                 # Typical coffee shop visit
        complexity_bias=0.3            # Lower complexity - more predictable
    ),
    
    'grocery_store': VenuePreset(
        name='Grocery Store',
        description='Functional shopping environment with task-focused interactions',
        flux_range=(0.1, 0.2),        # Moderate change - people come and go
        heat_range=(0.2, 0.5),        # Low to moderate volatility
        pace_range=(0.3, 0.5),        # Moderate pace - efficient movement
        count_range=(30, 80),          # Medium crowds
        max_hours=1.5,                 # Quick shopping trips
        complexity_bias=0.2            # Low complexity - functional interactions
    ),
    
    'nightclub': VenuePreset(
        name='Nightclub',
        description='High-energy social venue with intense, volatile dynamics',
        flux_range=(0.3, 0.7),        # High change rate - dynamic social mixing
        heat_range=(0.7, 0.95),       # High volatility - intense interactions
        pace_range=(0.6, 0.9),        # Fast pace - rapid social changes
        count_range=(80, 200),         # Large crowds
        max_hours=6.0,                 # Long night out
        complexity_bias=0.8            # High complexity - dramatic social dynamics
    ),
    
    'public_event': VenuePreset(
        name='Public Event',
        description='Large gathering with diverse groups and complex social flows',
        flux_range=(0.2, 0.5),        # Variable change rate - depends on event phase
        heat_range=(0.5, 0.85),       # Moderate to high volatility
        pace_range=(0.4, 0.7),        # Variable pace - event-dependent
        count_range=(100, 500),        # Large crowds
        max_hours=4.0,                 # Event duration
        complexity_bias=0.7            # High complexity - diverse social dynamics
    ),
    
    'office_party': VenuePreset(
        name='Office Party',
        description='Professional social gathering with constrained but evolving dynamics',
        flux_range=(0.1, 0.3),        # Moderate change - professional constraints
        heat_range=(0.4, 0.7),        # Moderate volatility - social but professional
        pace_range=(0.3, 0.6),        # Moderate pace - measured interactions
        count_range=(20, 100),         # Department to company size
        max_hours=3.0,                 # After-work event
        complexity_bias=0.5            # Moderate complexity - professional dynamics
    ),
    
    'house_party': VenuePreset(
        name='House Party',
        description='Intimate social gathering with personal dynamics and group formation',
        flux_range=(0.15, 0.4),       # Moderate change - friend groups mixing
        heat_range=(0.5, 0.8),        # Moderate to high volatility - personal dynamics
        pace_range=(0.4, 0.7),        # Moderate to fast pace - social energy
        count_range=(15, 60),          # Small to medium intimate groups
        max_hours=5.0,                 # Party duration
        complexity_bias=0.6            # Moderate-high complexity - personal dynamics
    ),
    
    'conference': VenuePreset(
        name='Conference',
        description='Professional networking with structured but dynamic interactions',
        flux_range=(0.2, 0.4),        # Moderate change - networking phases
        heat_range=(0.3, 0.6),        # Moderate volatility - professional networking
        pace_range=(0.5, 0.8),        # Fast pace - time-limited interactions
        count_range=(50, 300),         # Conference size
        max_hours=8.0,                 # Full day event
        complexity_bias=0.4            # Moderate complexity - structured networking
    ),
    
    'festival': VenuePreset(
        name='Festival',
        description='Large outdoor event with diverse activities and crowd flows',
        flux_range=(0.3, 0.6),        # High change - people moving between activities
        heat_range=(0.6, 0.9),        # High volatility - diverse crowd energy
        pace_range=(0.5, 0.8),        # Fast pace - festival energy
        count_range=(200, 1000),       # Large festival crowds
        max_hours=12.0,                # All-day event
        complexity_bias=0.9            # Very high complexity - maximum social dynamics
    ),
    
    'dramatic': VenuePreset(
        name='Dramatic Scenario',
        description='Artificially tuned for maximum complexity and double crossovers',
        flux_range=(0.001, 0.05),     # Very low flux - balanced initial conditions
        heat_range=(0.85, 0.99),      # Very high heat - maximum volatility
        pace_range=(0.1, 0.3),        # Slow pace - allows complex dynamics to develop
        count_range=(200, 500),        # Large populations - sustain longer dynamics
        max_hours=3.0,                 # Enough time for complex patterns
        complexity_bias=1.0            # Maximum complexity bias
    )
}


def get_venue_preset(venue_name: str) -> Optional[VenuePreset]:
    """
    Get a venue preset by name.
    
    Args:
        venue_name: Name of the venue preset
        
    Returns:
        VenuePreset object or None if not found
    """
    return VENUE_PRESETS.get(venue_name.lower())


def list_venues() -> List[str]:
    """
    Get a list of all available venue names.
    
    Returns:
        List of venue preset names
    """
    return list(VENUE_PRESETS.keys())


def create_dramatic_scenario(seed: Optional[int] = None) -> Dict[str, float]:
    """
    Create parameters specifically tuned for dramatic, complex scenarios.
    
    This function uses insights from successful double crossover scenarios
    to generate parameters likely to produce complex dynamics.
    
    Args:
        seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary of simulation parameters optimized for drama
    """
    dramatic_preset = VENUE_PRESETS['dramatic']
    return dramatic_preset.generate_parameters(seed)


def analyze_scenario_potential(flux: float, heat: float, pace: float, count: float) -> Dict[str, float]:
    """
    Analyze the potential for complex dynamics based on parameters.
    
    Args:
        flux: Rate of change parameter
        heat: Volatility parameter
        pace: Speed parameter
        count: Population size
        
    Returns:
        Dictionary with complexity metrics
    """
    # Factors that contribute to complex dynamics based on observed patterns
    
    # Low flux (balanced initial conditions) increases crossover potential
    flux_factor = max(0.0, 1.0 - abs(flux - 0.02) * 10.0)  # Optimal around 0.02, scale by 10
    
    # High heat increases volatility and crossover potential
    heat_factor = heat ** 2  # Quadratic scaling favors high heat
    
    # Moderate pace allows complex dynamics to develop
    pace_factor = max(0.0, 1.0 - abs(pace - 0.2) * 2.5)  # Optimal around 0.2, scale by 2.5
    
    # Larger populations can sustain longer dynamics
    count_factor = min(1.0, count / 300.0)  # Scale up to 300
    
    # Combined complexity score
    complexity_score = (flux_factor * 0.3 + heat_factor * 0.4 + 
                       pace_factor * 0.2 + count_factor * 0.1)
    
    # Normalize complexity score to 0-1 range
    complexity_score = min(1.0, complexity_score)
    
    # Probability estimates based on observed patterns
    double_crossover_prob = min(1.0, complexity_score * 0.3)  # Conservative estimate
    single_crossover_prob = min(1.0, complexity_score * 0.6 + 0.2)  # More likely
    annihilation_prob = max(0.0, 1.0 - complexity_score * 0.5)  # Decreases with complexity
    
    return {
        'complexity_score': complexity_score,
        'double_crossover_probability': double_crossover_prob,
        'single_crossover_probability': single_crossover_prob,
        'annihilation_probability': annihilation_prob,
        'flux_factor': flux_factor,
        'heat_factor': heat_factor,
        'pace_factor': pace_factor,
        'count_factor': count_factor
    }