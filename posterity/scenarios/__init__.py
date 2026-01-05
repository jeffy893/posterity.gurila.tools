"""
Scenario presets for posterity.gurila.tools tactical simulation engine.

This module provides predefined scenarios for different real-world venues
and situations, with parameters tuned to generate interesting dynamics.

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

from .venue_presets import VenuePreset, get_venue_preset, list_venues, create_dramatic_scenario, analyze_scenario_potential

__all__ = ['VenuePreset', 'get_venue_preset', 'list_venues', 'create_dramatic_scenario', 'analyze_scenario_potential']

__author__ = "Jefferson Richards"
__email__ = "jefferson@richards.plus"
__version__ = "1.0.0"