import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from typing import List, Dict #typing
from dataclasses import dataclass

@dataclass
class OneHealthParamters:
    """
    Rift valley fever spread in County X in Kenya
    """
    # Human population
    human_population: int = 10000
    human_birth_rates: float = 0.001
    human_death_rates: float = 0.015
    human_recovery_rate: float = 0.1
    # Livestock population
    livestock_population: int = 15000
    livestock_birth_rate: float = 0.001
    livestock_death_rate: float = 0.05
    livestock_recovery_rate: float = 0.8
    # Mosquitoes population
    mosquitoes_population: int = 500000
    mosquitoes_birth_rate: float = 0.02
    mosquitoes_death_rate: float = 0.05
    # Transmission rates
    livestock_to_mosquitoes: float = 0.00003
    mosquitoes_to_livestock: float = 0.00004
    mosquitoes_to_human: float = 0.0002
    livestock_to_human: float = 0.0001
    # Environment factors
    rainfall_factor: float = 1.0
    base_mosquito_breeding: float = 0.01
    # Intervention parameters
    livestock_vaccination_rate: float = 0.0
    mosquito_control_rate: float = 0.0

