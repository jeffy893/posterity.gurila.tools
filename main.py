#!/usr/bin/env python3
"""
Command-line interface for posterity.gurila.tools tactical simulation engine.

This script provides a CLI for running tactical simulations and getting
strategic recommendations based on the 3-7-12 framework.

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

import argparse
import sys
from typing import Optional
import json
from pathlib import Path

from posterity.core.simulation import run_tactical_simulation
from posterity.analysis.tactics import TacticalBrain, analyze_grocery_store_scenario
from posterity.reporting import generate_simulation_report
from posterity.scenarios import get_venue_preset, list_venues, create_dramatic_scenario, analyze_scenario_potential


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Posterity Tactical Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.10 main.py --flux 0.5 --heat 0.8 --count 50
  python3.10 main.py --venue nightclub --seed 42
  python3.10 main.py --dramatic --seed 777
  python3.10 main.py --venue festival --analyze-complexity
  python3.10 main.py --grocery-store
  python3.10 main.py --flux 0.2 --heat 0.9 --count 100 --json

Available venues: cafe, grocery_store, nightclub, public_event, office_party, 
house_party, conference, festival, dramatic
        """
    )
    
    # Core simulation parameters
    parser.add_argument(
        "--flux", 
        type=float, 
        default=0.5,
        help="Rate of change parameter (0.0 to 1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--heat", 
        type=float, 
        default=0.5,
        help="Volatility of Markov chain (0.0 to 1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--pace", 
        type=float, 
        default=0.5,
        help="Speed of iteration (0.0 to 1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--count", 
        type=float, 
        default=50.0,
        help="Initial population size (default: 50.0)"
    )
    
    # Optional parameters
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducible results"
    )
    
    parser.add_argument(
        "--max-hours", 
        type=float, 
        default=1.5,
        help="Maximum simulation time in hours (default: 1.5)"
    )
    
    # Special scenarios
    parser.add_argument(
        "--grocery-store",
        action="store_true",
        help="Run the classic grocery store scenario (Low Heat, Low Pace)"
    )
    
    parser.add_argument(
        "--venue",
        type=str,
        choices=list_venues(),
        help=f"Use venue-specific preset: {', '.join(list_venues())}"
    )
    
    parser.add_argument(
        "--dramatic",
        action="store_true",
        help="Generate parameters optimized for complex, dramatic scenarios"
    )
    
    parser.add_argument(
        "--analyze-complexity",
        action="store_true",
        help="Analyze the complexity potential of the given parameters"
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed analysis"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only output the final recommendation"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating HTML/PDF report (only show console output)"
    )
    
    parser.add_argument(
        "--report-dir",
        type=str,
        help="Directory for report output (default: current directory)"
    )
    
    return parser


def validate_parameters(args: argparse.Namespace) -> bool:
    """Validate command-line parameters."""
    errors = []
    
    if not 0.0 <= args.flux <= 1.0:
        errors.append("Flux must be between 0.0 and 1.0")
    
    if not 0.0 <= args.heat <= 1.0:
        errors.append("Heat must be between 0.0 and 1.0")
    
    if not 0.0 <= args.pace <= 1.0:
        errors.append("Pace must be between 0.0 and 1.0")
    
    if args.count <= 0:
        errors.append("Count must be positive")
    
    if args.max_hours <= 0:
        errors.append("Max hours must be positive")
    
    if errors:
        print("Parameter validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return False
    
    return True


def run_simulation(args: argparse.Namespace) -> tuple:
    """Run the tactical simulation and return results."""
    if args.grocery_store:
        # Run grocery store scenario
        recommendation = analyze_grocery_store_scenario()
        
        # Create a simplified result for grocery store
        results = {
            "scenario": "grocery_store",
            "parameters": {
                "heat": 0.2,
                "pace": 0.3,
                "flux": 0.1,
                "count": 50.0
            },
            "recommendation": {
                "approach": recommendation.approach.value,
                "target_group": recommendation.target_group.value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "summary": str(recommendation)
            }
        }
        
        return results, None, recommendation
    
    elif args.venue:
        # Use venue preset
        preset = get_venue_preset(args.venue)
        if not preset:
            raise ValueError(f"Unknown venue: {args.venue}")
        
        # Generate parameters from preset
        venue_params = preset.generate_parameters(args.seed)
        
        if not args.quiet:
            print(f"Using {preset.name} venue preset:")
            print(f"  {preset.description}")
            print(f"Running tactical simulation...")
            if args.verbose:
                print(f"Generated parameters: {venue_params}")
        
        # Run simulation with venue parameters
        result = run_tactical_simulation(
            pace=venue_params['pace'],
            flux=venue_params['flux'],
            heat=venue_params['heat'],
            count=venue_params['count'],
            random_seed=args.seed,
            max_hours=venue_params['max_hours']
        )
        
        # Analyze with tactical brain
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(
            result, 
            original_heat=venue_params['heat'], 
            original_pace=venue_params['pace']
        )
        
        # Prepare results
        results = {
            "scenario": f"venue_{args.venue}",
            "venue_preset": preset.name,
            "venue_description": preset.description,
            "parameters": venue_params,
            "simulation": {
                "duration": result.duration,
                "termination_reason": result.termination_reason.value,
                "num_crossovers": result.num_crossovers,
                "winner": result.winner,
                "final_bison": result.final_bison,
                "final_cattle": result.final_cattle
            },
            "recommendation": {
                "approach": recommendation.approach.value,
                "target_group": recommendation.target_group.value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "summary": str(recommendation)
            }
        }
        
        return results, result, recommendation
    
    elif args.dramatic:
        # Generate dramatic scenario parameters
        dramatic_params = create_dramatic_scenario(args.seed)
        
        if not args.quiet:
            print("Generating dramatic scenario optimized for complex dynamics...")
            if args.verbose:
                print(f"Dramatic parameters: {dramatic_params}")
        
        # Run simulation with dramatic parameters
        result = run_tactical_simulation(
            pace=dramatic_params['pace'],
            flux=dramatic_params['flux'],
            heat=dramatic_params['heat'],
            count=dramatic_params['count'],
            random_seed=args.seed,
            max_hours=dramatic_params['max_hours']
        )
        
        # Analyze with tactical brain
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(
            result, 
            original_heat=dramatic_params['heat'], 
            original_pace=dramatic_params['pace']
        )
        
        # Prepare results
        results = {
            "scenario": "dramatic",
            "parameters": dramatic_params,
            "simulation": {
                "duration": result.duration,
                "termination_reason": result.termination_reason.value,
                "num_crossovers": result.num_crossovers,
                "winner": result.winner,
                "final_bison": result.final_bison,
                "final_cattle": result.final_cattle
            },
            "recommendation": {
                "approach": recommendation.approach.value,
                "target_group": recommendation.target_group.value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "summary": str(recommendation)
            }
        }
        
        return results, result, recommendation
    
    else:
        # Run custom simulation
        if not args.quiet:
            print("Running tactical simulation...")
            if args.verbose:
                print(f"Parameters: flux={args.flux}, heat={args.heat}, "
                      f"pace={args.pace}, count={args.count}")
        
        # Analyze complexity potential if requested
        if args.analyze_complexity:
            complexity_analysis = analyze_scenario_potential(
                args.flux, args.heat, args.pace, args.count
            )
            if not args.quiet:
                print("\nComplexity Analysis:")
                print(f"  Complexity Score: {complexity_analysis['complexity_score']:.2f}")
                print(f"  Double Crossover Probability: {complexity_analysis['double_crossover_probability']:.1%}")
                print(f"  Single Crossover Probability: {complexity_analysis['single_crossover_probability']:.1%}")
                print(f"  Annihilation Probability: {complexity_analysis['annihilation_probability']:.1%}")
                print()
        
        # Run simulation
        result = run_tactical_simulation(
            pace=args.pace,
            flux=args.flux,
            heat=args.heat,
            count=args.count,
            random_seed=args.seed,
            max_hours=args.max_hours
        )
        
        # Analyze with tactical brain
        brain = TacticalBrain()
        recommendation = brain.analyze_simulation(
            result, 
            original_heat=args.heat, 
            original_pace=args.pace
        )
        
        # Prepare results
        results = {
            "scenario": "custom",
            "parameters": {
                "flux": args.flux,
                "heat": args.heat,
                "pace": args.pace,
                "count": args.count,
                "seed": args.seed,
                "max_hours": args.max_hours
            },
            "simulation": {
                "duration": result.duration,
                "termination_reason": result.termination_reason.value,
                "num_crossovers": result.num_crossovers,
                "winner": result.winner,
                "final_bison": result.final_bison,
                "final_cattle": result.final_cattle
            },
            "recommendation": {
                "approach": recommendation.approach.value,
                "target_group": recommendation.target_group.value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "summary": str(recommendation)
            }
        }
        
        # Add complexity analysis to results if requested
        if args.analyze_complexity:
            results["complexity_analysis"] = analyze_scenario_potential(
                args.flux, args.heat, args.pace, args.count
            )
        
        return results, result, recommendation


def format_output(results: dict, args: argparse.Namespace) -> str:
    """Format the output based on the requested format."""
    if args.json:
        return json.dumps(results, indent=2)
    
    elif args.quiet:
        return results["recommendation"]["summary"]
    
    else:
        # Standard human-readable format
        output = []
        
        # Header
        output.append("=" * 60)
        output.append("POSTERITY TACTICAL SIMULATION RESULTS")
        output.append("=" * 60)
        
        # Scenario info
        if results["scenario"] == "grocery_store":
            output.append("\nScenario: Grocery Store (Low Heat, Low Pace)")
        elif results["scenario"].startswith("venue_"):
            venue_name = results.get("venue_preset", "Unknown Venue")
            venue_desc = results.get("venue_description", "")
            output.append(f"\nScenario: {venue_name}")
            if venue_desc:
                output.append(f"Description: {venue_desc}")
        elif results["scenario"] == "dramatic":
            output.append(f"\nScenario: Dramatic (Optimized for Complex Dynamics)")
        else:
            output.append(f"\nScenario: Custom Simulation")
        
        # Parameters
        output.append("\nParameters:")
        params = results["parameters"]
        for key, value in params.items():
            if value is not None:
                output.append(f"  {key.capitalize()}: {value}")
        
        # Simulation results (if available)
        if "simulation" in results:
            sim = results["simulation"]
            output.append(f"\nSimulation Results:")
            output.append(f"  Duration: {sim['duration']:.2f} seconds")
            output.append(f"  Termination: {sim['termination_reason'].replace('_', ' ').title()}")
            output.append(f"  Crossovers: {sim['num_crossovers']}")
            if sim['winner']:
                output.append(f"  Winner: {sim['winner'].title()}")
            output.append(f"  Final Populations: Bison={sim['final_bison']:.1f}, Cattle={sim['final_cattle']:.1f}")
        
        # Recommendation
        rec = results["recommendation"]
        output.append(f"\nTactical Recommendation:")
        output.append(f"  {rec['summary']}")
        output.append(f"  Confidence: {rec['confidence']:.1%}")
        
        if args.verbose:
            output.append(f"\nReasoning:")
            output.append(f"  {rec['reasoning']}")
            
            # Add complexity analysis if available
            if "complexity_analysis" in results:
                comp = results["complexity_analysis"]
                output.append(f"\nComplexity Analysis:")
                output.append(f"  Complexity Score: {comp['complexity_score']:.2f}")
                output.append(f"  Double Crossover Probability: {comp['double_crossover_probability']:.1%}")
                output.append(f"  Single Crossover Probability: {comp['single_crossover_probability']:.1%}")
                output.append(f"  Annihilation Probability: {comp['annihilation_probability']:.1%}")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate parameters
    if not validate_parameters(args):
        return 1
    
    try:
        # Run simulation
        results, simulation_result, recommendation = run_simulation(args)
        
        # Format and output results
        output = format_output(results, args)
        print(output)
        
        # Generate report if requested
        if not args.no_report:
            if not args.quiet:
                print("\nGenerating simulation report...")
            
            try:
                # Determine output directory
                output_dir = Path(args.report_dir) if args.report_dir else Path.cwd()
                
                # Generate report
                report_dir = generate_simulation_report(
                    parameters=results["parameters"],
                    simulation_result=simulation_result,
                    recommendation=recommendation,
                    trajectory=simulation_result.trajectory if simulation_result else None,
                    alpha_coeffs=simulation_result.alpha_coefficients if simulation_result else None,
                    beta_coeffs=simulation_result.beta_coefficients if simulation_result else None,
                    output_dir=output_dir
                )
                
                if not args.quiet:
                    print(f"Report generated in: {report_dir}")
                    print(f"  - HTML: {report_dir / 'report.html'}")
                    print(f"  - PDF: {report_dir / 'report.pdf'}")
                    if simulation_result and (report_dir / 'population_trajectory.png').exists():
                        print(f"  - Charts: {report_dir / 'population_trajectory.png'}")
                        if (report_dir / 'coefficient_evolution.png').exists():
                            print(f"            {report_dir / 'coefficient_evolution.png'}")
                    print(f"  - Data: {report_dir / 'raw_data.json'}")
                
            except Exception as e:
                print(f"Warning: Could not generate report: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.", file=sys.stderr)
        return 130
    
    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())