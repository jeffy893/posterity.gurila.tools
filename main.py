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

from posterity.core.simulation import run_tactical_simulation
from posterity.analysis.tactics import TacticalBrain, analyze_grocery_store_scenario


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Posterity Tactical Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --flux 0.5 --heat 0.8 --count 50
  python main.py --flux 0.3 --heat 0.4 --pace 0.6 --count 75 --seed 42
  python main.py --grocery-store
  python main.py --flux 0.2 --heat 0.9 --count 100 --json
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


def run_simulation(args: argparse.Namespace) -> dict:
    """Run the tactical simulation and return results."""
    if args.grocery_store:
        # Run grocery store scenario
        recommendation = analyze_grocery_store_scenario()
        
        # Create a simplified result for grocery store
        return {
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
    
    else:
        # Run custom simulation
        if not args.quiet:
            print("Running tactical simulation...")
            if args.verbose:
                print(f"Parameters: flux={args.flux}, heat={args.heat}, "
                      f"pace={args.pace}, count={args.count}")
        
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
        return {
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
        results = run_simulation(args)
        
        # Format and output results
        output = format_output(results, args)
        print(output)
        
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