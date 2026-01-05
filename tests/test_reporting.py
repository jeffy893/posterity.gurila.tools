"""
Tests for the reporting module.

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
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing anything else that uses matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

from posterity.core.simulation import run_tactical_simulation
from posterity.analysis.tactics import TacticalBrain
from posterity.reporting import ReportGenerator, generate_simulation_report


class TestReportGenerator(unittest.TestCase):
    """Test cases for the ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = ReportGenerator(self.temp_dir)
        
        # Create test simulation data
        self.result = run_tactical_simulation(
            pace=0.5,
            flux=0.3,
            heat=0.4,
            count=50.0,
            random_seed=42
        )
        
        self.brain = TacticalBrain()
        self.recommendation = self.brain.analyze_simulation(
            self.result,
            original_heat=0.4,
            original_pace=0.5
        )
        
        self.parameters = {
            "flux": 0.3,
            "heat": 0.4,
            "pace": 0.5,
            "count": 50.0,
            "seed": 42
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReportGenerator initialization."""
        self.assertEqual(self.generator.output_dir, self.temp_dir)
        self.assertIsNotNone(self.generator.template_env)
        self.assertIn('bison', self.generator.colors)
        self.assertIn('cattle', self.generator.colors)
    
    def test_create_population_chart(self):
        """Test population chart creation."""
        output_path = self.temp_dir / 'test_population.png'
        
        self.generator.create_population_chart(
            self.result.trajectory,
            output_path
        )
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 1000)  # Should be a reasonable file size
    
    def test_create_coefficient_chart(self):
        """Test coefficient chart creation."""
        output_path = self.temp_dir / 'test_coefficients.png'
        
        self.generator.create_coefficient_chart(
            self.result.trajectory,
            self.result.alpha_coefficients,
            self.result.beta_coefficients,
            output_path
        )
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 1000)
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        output_path = self.temp_dir / 'test_report.html'
        
        self.generator.generate_html_report(
            self.parameters,
            self.result,
            self.recommendation,
            output_path
        )
        
        self.assertTrue(output_path.exists())
        
        # Check content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('Posterity Tactical Simulation', content)
        self.assertIn(str(self.parameters['flux']), content)
        self.assertIn(self.recommendation.approach.value, content)
    
    def test_generate_full_report(self):
        """Test complete report generation."""
        report_dir = self.generator.generate_full_report(
            self.parameters,
            self.result,
            self.recommendation,
            self.result.trajectory,
            self.result.alpha_coefficients,
            self.result.beta_coefficients,
            "test_report"
        )
        
        # Check that it's in the simulation_reports subdirectory
        expected_path = self.temp_dir / "simulation_reports" / "test_report"
        self.assertEqual(report_dir, expected_path)
        
        self.assertTrue(report_dir.exists())
        self.assertTrue((report_dir / 'report.html').exists())
        self.assertTrue((report_dir / 'population_trajectory.png').exists())
        self.assertTrue((report_dir / 'coefficient_evolution.png').exists())
        self.assertTrue((report_dir / 'raw_data.json').exists())
        
        # Check if PDF was generated (may fail if weasyprint dependencies missing)
        pdf_path = report_dir / 'report.pdf'
        if pdf_path.exists():
            self.assertGreater(pdf_path.stat().st_size, 1000)


class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function for report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_simulation_report(self):
        """Test the convenience function."""
        # Run simulation
        result = run_tactical_simulation(
            pace=0.6,
            flux=0.4,
            heat=0.3,
            count=75.0,
            random_seed=123
        )
        
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(
            result,
            original_heat=0.3,
            original_pace=0.6
        )
        
        parameters = {
            "flux": 0.4,
            "heat": 0.3,
            "pace": 0.6,
            "count": 75.0,
            "seed": 123
        }
        
        # Generate report
        report_dir = generate_simulation_report(
            parameters,
            result,
            recommendation,
            result.trajectory,
            result.alpha_coefficients,
            result.beta_coefficients,
            self.temp_dir,
            "convenience_test"
        )
        
        self.assertTrue(report_dir.exists())
        self.assertTrue((report_dir / 'report.html').exists())
        self.assertTrue((report_dir / 'raw_data.json').exists())
        
        # Check that it's in the simulation_reports subdirectory
        self.assertTrue("simulation_reports" in str(report_dir))


class TestGroceryStoreReporting(unittest.TestCase):
    """Test reporting for grocery store scenario."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_grocery_store_report(self):
        """Test report generation for grocery store scenario."""
        from posterity.analysis.tactics import analyze_grocery_store_scenario
        
        recommendation = analyze_grocery_store_scenario()
        parameters = {
            "heat": 0.2,
            "pace": 0.3,
            "flux": 0.1,
            "count": 50.0,
            "scenario": "grocery_store"
        }
        
        # Generate report without simulation result
        report_dir = generate_simulation_report(
            parameters,
            None,  # No simulation result for grocery store
            recommendation,
            output_dir=self.temp_dir,
            report_name="grocery_store_test"
        )
        
        self.assertTrue(report_dir.exists())
        self.assertTrue((report_dir / 'report.html').exists())
        self.assertTrue((report_dir / 'raw_data.json').exists())
        
        # Check that it's in the simulation_reports subdirectory
        self.assertTrue("simulation_reports" in str(report_dir))
        
        # Check HTML content
        with open(report_dir / 'report.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('Posterity Tactical Simulation', content)
        self.assertIn(recommendation.approach.value, content)


if __name__ == '__main__':
    unittest.main()