"""
Report generation module for tactical simulation results.

This module provides comprehensive reporting capabilities including HTML reports,
PNG charts, and PDF generation with proper margin handling.

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

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import jinja2
import weasyprint
from weasyprint import HTML, CSS

from ..core.simulation import SimulationResult
from ..analysis.tactics import TacticalRecommendation


class ReportGenerator:
    """
    Generates comprehensive reports for tactical simulation results.
    
    Creates HTML reports, PNG charts, and PDF documents with proper formatting
    and margin handling for professional presentation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Base directory for report output. If None, uses current directory.
        """
        self.output_dir = output_dir or Path.cwd()
        self.template_env = self._setup_jinja_environment()
        
        # Chart styling
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.colors = {
            'bison': '#2E8B57',      # Sea Green
            'cattle': '#CD853F',     # Peru
            'background': '#F8F9FA', # Light Gray
            'grid': '#E9ECEF',       # Lighter Gray
            'text': '#212529'        # Dark Gray
        }
    
    def _setup_jinja_environment(self) -> jinja2.Environment:
        """Setup Jinja2 template environment."""
        template_dir = Path(__file__).parent / 'templates'
        template_dir.mkdir(exist_ok=True)
        
        # Create template if it doesn't exist
        self._create_html_template(template_dir)
        
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def _create_html_template(self, template_dir: Path) -> None:
        """Create the HTML template file."""
        template_path = template_dir / 'report_template.html'
        
        if not template_path.exists():
            template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Posterity Tactical Simulation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #2E8B57, #228B22);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #2E8B57;
            border-bottom: 2px solid #2E8B57;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .parameters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .parameter-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E8B57;
        }
        .parameter-card .label {
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        .parameter-card .value {
            font-size: 1.4em;
            color: #2E8B57;
            margin-top: 5px;
        }
        .recommendation {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border: 2px solid #2196f3;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
        }
        .recommendation h3 {
            color: #1976d2;
            margin-top: 0;
        }
        .recommendation .summary {
            font-size: 1.3em;
            font-weight: bold;
            color: #1565c0;
            margin-bottom: 15px;
        }
        .confidence-bar {
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.3s ease;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }
        .timestamp {
            color: #888;
            font-size: 0.9em;
        }
        @media print {
            body { background: white; }
            .section { box-shadow: none; border: 1px solid #ddd; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Posterity Tactical Simulation</h1>
        <div class="subtitle">Strategic Analysis Report</div>
        <div class="timestamp">Generated: {{ timestamp }}</div>
    </div>

    <div class="section">
        <h2>Simulation Parameters</h2>
        <div class="parameters-grid">
            {% for param, value in parameters.items() %}
            <div class="parameter-card">
                <div class="label">{{ param.replace('_', ' ').title() }}</div>
                <div class="value">{{ value }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    {% if simulation_results %}
    <div class="section">
        <h2>Simulation Results</h2>
        <div class="parameters-grid">
            <div class="parameter-card">
                <div class="label">Duration</div>
                <div class="value">{{ "%.2f"|format(simulation_results.duration) }}s</div>
            </div>
            <div class="parameter-card">
                <div class="label">Termination</div>
                <div class="value">{{ simulation_results.termination_reason.replace('_', ' ').title() }}</div>
            </div>
            <div class="parameter-card">
                <div class="label">Crossovers</div>
                <div class="value">{{ simulation_results.num_crossovers }}</div>
            </div>
            {% if simulation_results.winner %}
            <div class="parameter-card">
                <div class="label">Winner</div>
                <div class="value">{{ simulation_results.winner.title() }}</div>
            </div>
            {% endif %}
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>Population Dynamics</h2>
        <div class="chart-container">
            <img src="population_trajectory.png" alt="Population Trajectory Chart">
        </div>
    </div>

    <div class="section">
        <h2>Coefficient Evolution</h2>
        <div class="chart-container">
            <img src="coefficient_evolution.png" alt="Coefficient Evolution Chart">
        </div>
    </div>

    <div class="section">
        <h2>Tactical Recommendation</h2>
        <div class="recommendation">
            <h3>Strategic Advice</h3>
            <div class="summary">{{ recommendation.summary }}</div>
            <p><strong>Approach:</strong> {{ recommendation.approach.title() }}</p>
            <p><strong>Target Group:</strong> {{ recommendation.target_group }}</p>
            <p><strong>Confidence:</strong> {{ "%.1f"|format(recommendation.confidence * 100) }}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {{ recommendation.confidence * 100 }}%"></div>
            </div>
            <p><strong>Reasoning:</strong> {{ recommendation.reasoning }}</p>
        </div>
    </div>

    <div class="footer">
        <p>Generated by Posterity.gurila.tools Tactical Simulation Engine</p>
        <p>Contact: jefferson@richards.plus | Licensed under GPL v3</p>
    </div>
</body>
</html>'''
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
    
    def create_population_chart(
        self, 
        trajectory: NDArray[np.float64], 
        output_path: Path,
        title: str = "Population Dynamics Over Time"
    ) -> None:
        """
        Create a population trajectory chart.
        
        Args:
            trajectory: Array of shape (n_steps, 3) with [time, bison, cattle]
            output_path: Path to save the PNG file
            title: Chart title
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        time = trajectory[:, 0]
        bison = trajectory[:, 1]
        cattle = trajectory[:, 2]
        
        # Plot population curves
        ax.plot(time, bison, 
                color=self.colors['bison'], 
                linewidth=3, 
                label='Bison (Active/High Morale)',
                alpha=0.8)
        
        ax.plot(time, cattle, 
                color=self.colors['cattle'], 
                linewidth=3, 
                label='Cattle (Passive/Low Morale)',
                alpha=0.8)
        
        # Styling
        ax.set_xlabel('Time (simulation units)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(fontsize=11, framealpha=0.9)
        
        # Set background color
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor('white')
        
        # Tight layout to fit within margins
        plt.tight_layout(pad=2.0)
        
        # Save with high DPI for quality
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor=self.colors['background'],
                   edgecolor='none')
        plt.close(fig)
    
    def create_coefficient_chart(
        self,
        trajectory: NDArray[np.float64],
        alpha_coeffs: List[float],
        beta_coeffs: List[float],
        output_path: Path,
        title: str = "Markov Chain Coefficient Evolution"
    ) -> None:
        """
        Create a coefficient evolution chart.
        
        Args:
            trajectory: Array of shape (n_steps, 3) with [time, bison, cattle]
            alpha_coeffs: List of alpha coefficients over time
            beta_coeffs: List of beta coefficients over time
            output_path: Path to save the PNG file
            title: Chart title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        time = trajectory[:len(alpha_coeffs), 0]  # Ensure matching lengths
        
        # Alpha coefficients
        ax1.plot(time, alpha_coeffs[:len(time)], 
                color='#FF6B6B', 
                linewidth=2, 
                label='Alpha (Bison Effectiveness)',
                alpha=0.8)
        
        ax1.set_ylabel('Alpha Coefficient', fontsize=11, fontweight='bold')
        ax1.set_title('Alpha Coefficient Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_facecolor('white')
        
        # Beta coefficients
        ax2.plot(time, beta_coeffs[:len(time)], 
                color='#4ECDC4', 
                linewidth=2, 
                label='Beta (Cattle Effectiveness)',
                alpha=0.8)
        
        ax2.set_xlabel('Time (simulation units)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Beta Coefficient', fontsize=11, fontweight='bold')
        ax2.set_title('Beta Coefficient Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_facecolor('white')
        
        # Overall styling
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        fig.patch.set_facecolor(self.colors['background'])
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.93)
        
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor=self.colors['background'],
                   edgecolor='none')
        plt.close(fig)
    
    def generate_html_report(
        self,
        parameters: Dict[str, Any],
        simulation_result: Optional[SimulationResult],
        recommendation: TacticalRecommendation,
        output_path: Path
    ) -> None:
        """
        Generate HTML report.
        
        Args:
            parameters: Simulation parameters
            simulation_result: Simulation results (None for grocery store scenario)
            recommendation: Tactical recommendation
            output_path: Path to save HTML file
        """
        template = self.template_env.get_template('report_template.html')
        
        # Prepare template data
        template_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': parameters,
            'simulation_results': simulation_result,
            'recommendation': {
                'approach': recommendation.approach.value,
                'target_group': recommendation.target_group.value,
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'summary': str(recommendation)
            }
        }
        
        # Render and save
        html_content = template.render(**template_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_pdf_report(
        self,
        html_path: Path,
        output_path: Path
    ) -> None:
        """
        Generate PDF report from HTML with proper margin handling.
        
        Args:
            html_path: Path to HTML file
            output_path: Path to save PDF file
        """
        # CSS for PDF formatting with proper margins
        pdf_css = CSS(string='''
            @page {
                size: A4;
                margin: 2cm 1.5cm 2cm 1.5cm;
            }
            
            body {
                font-size: 11pt;
                line-height: 1.4;
            }
            
            .header {
                margin-bottom: 20px;
            }
            
            .header h1 {
                font-size: 24pt;
                margin-bottom: 10px;
            }
            
            .section {
                margin-bottom: 20px;
                page-break-inside: avoid;
            }
            
            .section h2 {
                font-size: 16pt;
                margin-bottom: 10px;
            }
            
            .parameters-grid {
                display: block;
            }
            
            .parameter-card {
                display: inline-block;
                width: 45%;
                margin: 5px;
                vertical-align: top;
            }
            
            .chart-container img {
                max-width: 100%;
                height: auto;
                page-break-inside: avoid;
            }
            
            .recommendation {
                page-break-inside: avoid;
            }
        ''')
        
        # Generate PDF
        html_doc = HTML(filename=str(html_path))
        html_doc.write_pdf(str(output_path), stylesheets=[pdf_css])
    
    def generate_full_report(
        self,
        parameters: Dict[str, Any],
        simulation_result: Optional[SimulationResult],
        recommendation: TacticalRecommendation,
        trajectory: Optional[NDArray[np.float64]] = None,
        alpha_coeffs: Optional[List[float]] = None,
        beta_coeffs: Optional[List[float]] = None,
        report_name: Optional[str] = None
    ) -> Path:
        """
        Generate a complete report with HTML, PNG charts, and PDF.
        
        Args:
            parameters: Simulation parameters
            simulation_result: Simulation results
            recommendation: Tactical recommendation
            trajectory: Population trajectory data
            alpha_coeffs: Alpha coefficient history
            beta_coeffs: Beta coefficient history
            report_name: Custom report name (auto-generated if None)
            
        Returns:
            Path to the report directory
        """
        # Create simulation_reports directory at root level
        reports_root = self.output_dir / "simulation_reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        
        # Create individual report directory
        if report_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"simulation_report_{timestamp}"
        
        report_dir = reports_root / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts if trajectory data is available
        if trajectory is not None:
            self.create_population_chart(
                trajectory, 
                report_dir / 'population_trajectory.png'
            )
            
            if alpha_coeffs and beta_coeffs:
                self.create_coefficient_chart(
                    trajectory,
                    alpha_coeffs,
                    beta_coeffs,
                    report_dir / 'coefficient_evolution.png'
                )
        
        # Generate HTML report
        html_path = report_dir / 'report.html'
        self.generate_html_report(
            parameters,
            simulation_result,
            recommendation,
            html_path
        )
        
        # Generate PDF report
        pdf_path = report_dir / 'report.pdf'
        try:
            self.generate_pdf_report(html_path, pdf_path)
        except Exception as e:
            print(f"Warning: Could not generate PDF report: {e}")
            print("HTML report generated successfully.")
        
        # Save raw data as JSON
        json_path = report_dir / 'raw_data.json'
        raw_data = {
            'parameters': parameters,
            'simulation_result': {
                'duration': simulation_result.duration if simulation_result else None,
                'termination_reason': simulation_result.termination_reason.value if simulation_result else None,
                'num_crossovers': simulation_result.num_crossovers if simulation_result else None,
                'winner': simulation_result.winner if simulation_result else None,
                'final_bison': simulation_result.final_bison if simulation_result else None,
                'final_cattle': simulation_result.final_cattle if simulation_result else None,
            } if simulation_result else None,
            'recommendation': {
                'approach': recommendation.approach.value,
                'target_group': recommendation.target_group.value,
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'summary': str(recommendation)
            },
            'trajectory': trajectory.tolist() if trajectory is not None else None,
            'coefficients': {
                'alpha': alpha_coeffs,
                'beta': beta_coeffs
            } if alpha_coeffs and beta_coeffs else None
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2)
        
        return report_dir


def generate_simulation_report(
    parameters: Dict[str, Any],
    simulation_result: Optional[SimulationResult],
    recommendation: TacticalRecommendation,
    trajectory: Optional[NDArray[np.float64]] = None,
    alpha_coeffs: Optional[List[float]] = None,
    beta_coeffs: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
    report_name: Optional[str] = None
) -> Path:
    """
    Convenience function to generate a complete simulation report.
    
    Args:
        parameters: Simulation parameters
        simulation_result: Simulation results
        recommendation: Tactical recommendation
        trajectory: Population trajectory data
        alpha_coeffs: Alpha coefficient history
        beta_coeffs: Beta coefficient history
        output_dir: Output directory (defaults to current directory)
        report_name: Custom report name
        
    Returns:
        Path to the generated report directory
    """
    generator = ReportGenerator(output_dir)
    return generator.generate_full_report(
        parameters,
        simulation_result,
        recommendation,
        trajectory,
        alpha_coeffs,
        beta_coeffs,
        report_name
    )