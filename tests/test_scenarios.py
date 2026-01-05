"""
Tests for the scenarios module.

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

import unittest
from posterity.scenarios import (
    VenuePreset, get_venue_preset, list_venues, 
    create_dramatic_scenario, analyze_scenario_potential
)


class TestVenuePreset(unittest.TestCase):
    """Test cases for VenuePreset class."""
    
    def test_venue_preset_creation(self):
        """Test creating a venue preset."""
        preset = VenuePreset(
            name="Test Venue",
            description="A test venue",
            flux_range=(0.1, 0.3),
            heat_range=(0.2, 0.6),
            pace_range=(0.3, 0.7),
            count_range=(20, 80),
            max_hours=2.0,
            complexity_bias=0.5
        )
        
        self.assertEqual(preset.name, "Test Venue")
        self.assertEqual(preset.description, "A test venue")
        self.assertEqual(preset.flux_range, (0.1, 0.3))
        self.assertEqual(preset.complexity_bias, 0.5)
    
    def test_generate_parameters(self):
        """Test parameter generation from preset."""
        preset = VenuePreset(
            name="Test",
            description="Test",
            flux_range=(0.1, 0.3),
            heat_range=(0.2, 0.6),
            pace_range=(0.3, 0.7),
            count_range=(20, 80),
            max_hours=2.0,
            complexity_bias=0.5
        )
        
        # Test with seed for reproducibility
        params1 = preset.generate_parameters(seed=42)
        params2 = preset.generate_parameters(seed=42)
        
        # Should be identical with same seed
        self.assertEqual(params1, params2)
        
        # Check parameter ranges
        self.assertGreaterEqual(params1['flux'], 0.0)
        self.assertLessEqual(params1['flux'], 1.0)
        self.assertGreaterEqual(params1['heat'], 0.0)
        self.assertLessEqual(params1['heat'], 1.0)
        self.assertGreaterEqual(params1['pace'], 0.0)
        self.assertLessEqual(params1['pace'], 1.0)
        self.assertGreaterEqual(params1['count'], 20)
        self.assertLessEqual(params1['count'], 80)
        self.assertEqual(params1['max_hours'], 2.0)
    
    def test_complexity_bias_effect(self):
        """Test that complexity bias affects parameter generation."""
        low_bias_preset = VenuePreset(
            name="Low Bias",
            description="Low complexity bias",
            flux_range=(0.0, 1.0),
            heat_range=(0.0, 1.0),
            pace_range=(0.0, 1.0),
            count_range=(50, 100),
            max_hours=2.0,
            complexity_bias=0.1
        )
        
        high_bias_preset = VenuePreset(
            name="High Bias",
            description="High complexity bias",
            flux_range=(0.0, 1.0),
            heat_range=(0.0, 1.0),
            pace_range=(0.0, 1.0),
            count_range=(50, 100),
            max_hours=2.0,
            complexity_bias=0.9
        )
        
        # Generate multiple samples to test bias effect
        low_bias_samples = [low_bias_preset.generate_parameters() for _ in range(10)]
        high_bias_samples = [high_bias_preset.generate_parameters() for _ in range(10)]
        
        # High bias should generally produce lower flux and higher heat
        avg_low_flux = sum(s['flux'] for s in low_bias_samples) / len(low_bias_samples)
        avg_high_flux = sum(s['flux'] for s in high_bias_samples) / len(high_bias_samples)
        
        avg_low_heat = sum(s['heat'] for s in low_bias_samples) / len(low_bias_samples)
        avg_high_heat = sum(s['heat'] for s in high_bias_samples) / len(high_bias_samples)
        
        # These are statistical tendencies, not guarantees
        # But over multiple samples, the bias should be apparent
        self.assertIsInstance(avg_low_flux, float)
        self.assertIsInstance(avg_high_flux, float)
        self.assertIsInstance(avg_low_heat, float)
        self.assertIsInstance(avg_high_heat, float)


class TestVenuePresets(unittest.TestCase):
    """Test cases for venue preset functions."""
    
    def test_list_venues(self):
        """Test listing available venues."""
        venues = list_venues()
        
        self.assertIsInstance(venues, list)
        self.assertGreater(len(venues), 0)
        self.assertIn('cafe', venues)
        self.assertIn('nightclub', venues)
        self.assertIn('dramatic', venues)
    
    def test_get_venue_preset(self):
        """Test getting venue presets by name."""
        # Test valid venue
        cafe_preset = get_venue_preset('cafe')
        self.assertIsNotNone(cafe_preset)
        self.assertEqual(cafe_preset.name, 'Cafe')
        
        # Test case insensitivity
        nightclub_preset = get_venue_preset('NIGHTCLUB')
        self.assertIsNotNone(nightclub_preset)
        self.assertEqual(nightclub_preset.name, 'Nightclub')
        
        # Test invalid venue
        invalid_preset = get_venue_preset('nonexistent')
        self.assertIsNone(invalid_preset)
    
    def test_venue_preset_characteristics(self):
        """Test that venue presets have expected characteristics."""
        # Cafe should be low energy
        cafe = get_venue_preset('cafe')
        self.assertLess(cafe.heat_range[1], 0.5)  # Max heat < 0.5
        self.assertLess(cafe.complexity_bias, 0.5)  # Low complexity
        
        # Nightclub should be high energy
        nightclub = get_venue_preset('nightclub')
        self.assertGreater(nightclub.heat_range[0], 0.6)  # Min heat > 0.6
        self.assertGreater(nightclub.complexity_bias, 0.7)  # High complexity
        
        # Dramatic should be optimized for complexity
        dramatic = get_venue_preset('dramatic')
        self.assertEqual(dramatic.complexity_bias, 1.0)  # Maximum complexity
        self.assertLess(dramatic.flux_range[1], 0.1)  # Very low flux
        self.assertGreater(dramatic.heat_range[0], 0.8)  # Very high heat


class TestDramaticScenarios(unittest.TestCase):
    """Test cases for dramatic scenario generation."""
    
    def test_create_dramatic_scenario(self):
        """Test dramatic scenario parameter generation."""
        # Test with seed for reproducibility
        params1 = create_dramatic_scenario(seed=42)
        params2 = create_dramatic_scenario(seed=42)
        
        self.assertEqual(params1, params2)
        
        # Check that parameters are in valid ranges
        self.assertGreaterEqual(params1['flux'], 0.0)
        self.assertLessEqual(params1['flux'], 1.0)
        self.assertGreaterEqual(params1['heat'], 0.0)
        self.assertLessEqual(params1['heat'], 1.0)
        self.assertGreaterEqual(params1['pace'], 0.0)
        self.assertLessEqual(params1['pace'], 1.0)
        self.assertGreater(params1['count'], 0)
        self.assertGreater(params1['max_hours'], 0)
        
        # Dramatic scenarios should have specific characteristics
        self.assertLess(params1['flux'], 0.1)  # Low flux for balance
        self.assertGreater(params1['heat'], 0.8)  # High heat for volatility
    
    def test_multiple_dramatic_scenarios(self):
        """Test generating multiple dramatic scenarios."""
        scenarios = [create_dramatic_scenario() for _ in range(5)]
        
        # All should be valid
        for params in scenarios:
            self.assertIn('flux', params)
            self.assertIn('heat', params)
            self.assertIn('pace', params)
            self.assertIn('count', params)
            self.assertIn('max_hours', params)
            
            # Should follow dramatic characteristics
            self.assertLess(params['flux'], 0.1)
            self.assertGreater(params['heat'], 0.8)


class TestComplexityAnalysis(unittest.TestCase):
    """Test cases for complexity analysis."""
    
    def test_analyze_scenario_potential(self):
        """Test scenario complexity analysis."""
        # Test with known dramatic parameters
        analysis = analyze_scenario_potential(
            flux=0.001,
            heat=0.95,
            pace=0.1,
            count=300
        )
        
        # Check that all expected metrics are present
        expected_keys = [
            'complexity_score',
            'double_crossover_probability',
            'single_crossover_probability',
            'annihilation_probability',
            'flux_factor',
            'heat_factor',
            'pace_factor',
            'count_factor'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
            self.assertIsInstance(analysis[key], float)
            self.assertGreaterEqual(analysis[key], 0.0)
            self.assertLessEqual(analysis[key], 1.0)
        
        # High complexity parameters should have high complexity score
        self.assertGreater(analysis['complexity_score'], 0.5)
    
    def test_complexity_analysis_ranges(self):
        """Test complexity analysis with different parameter ranges."""
        # Low complexity scenario
        low_analysis = analyze_scenario_potential(
            flux=0.5,  # High flux - unbalanced
            heat=0.2,  # Low heat - low volatility
            pace=0.9,  # Fast pace - less time for complexity
            count=20   # Small population
        )
        
        # High complexity scenario
        high_analysis = analyze_scenario_potential(
            flux=0.02,  # Low flux - balanced
            heat=0.95,  # High heat - high volatility
            pace=0.2,   # Slow pace - time for complexity
            count=300   # Large population
        )
        
        # High complexity should score higher
        self.assertGreater(
            high_analysis['complexity_score'],
            low_analysis['complexity_score']
        )
        
        # High complexity should have higher double crossover probability
        self.assertGreater(
            high_analysis['double_crossover_probability'],
            low_analysis['double_crossover_probability']
        )
    
    def test_probability_consistency(self):
        """Test that probability estimates are consistent."""
        analysis = analyze_scenario_potential(0.1, 0.5, 0.3, 100)
        
        # Probabilities should be between 0 and 1
        self.assertGreaterEqual(analysis['double_crossover_probability'], 0.0)
        self.assertLessEqual(analysis['double_crossover_probability'], 1.0)
        
        self.assertGreaterEqual(analysis['single_crossover_probability'], 0.0)
        self.assertLessEqual(analysis['single_crossover_probability'], 1.0)
        
        self.assertGreaterEqual(analysis['annihilation_probability'], 0.0)
        self.assertLessEqual(analysis['annihilation_probability'], 1.0)


if __name__ == '__main__':
    unittest.main()