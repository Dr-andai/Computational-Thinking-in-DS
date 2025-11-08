import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from typing import List, Dict
from dataclasses import dataclass

from one_health import OneHealthSIR, OneHealthParamters

class OneHealthVisualizer:
    """Visualization class for the One Health SIR model"""
    
    def __init__(self, model: OneHealthSIR):
        self.model = model
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('One Health SIR Model: Rift Valley Fever in Kenya\n'
                         'Human-livestock-Mosquito Dynamics', fontsize=16, fontweight='bold')
        
    def create_initial_plot(self):
        """Create the initial static plot"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Human Population Dynamics
        ax1 = self.axes[0, 0]
        ax1.set_title('Human Population Dynamics')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Number of Individuals')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: livestock Population Dynamics
        ax2 = self.axes[0, 1]
        ax2.set_title('livestock Population Dynamics')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Number of Animals')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mosquito Population Dynamics
        ax3 = self.axes[1, 0]
        ax3.set_title('Mosquito Population Dynamics')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Number of Mosquitoes')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: One Health Overview (Network diagram)
        ax4 = self.axes[1, 1]
        ax4.set_title('One Health Transmission Network')
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)
        ax4.set_aspect('equal')
        ax4.axis('off')
        
        return ax1, ax2, ax3, ax4
    
    def update_network_diagram(self, ax, current_data):
        """Update the network diagram showing transmission pathways"""
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('One Health Transmission Network\n(Current State)')
        ax.axis('off')
        
        # Define node positions
        nodes = {
            'Humans': (2, 8),
            'livestock': (8, 8),
            'Mosquitoes': (5, 3),
            'Environment': (5, 6)
        }
        
        # Draw nodes with size proportional to infected population
        node_colors = {'Humans': 'lightcoral', 'livestock': 'lightgreen', 
                      'Mosquitoes': 'lightblue', 'Environment': 'lightyellow'}
        
        for node, (x, y) in nodes.items():
            if node == 'Humans':
                size = current_data['I_h'] / 100 + 1
                label = f'Humans\nInfected: {int(current_data["I_h"])}'
            elif node == 'livestock':
                size = current_data['I_L'] / 50 + 1
                label = f'livestock\nInfected: {int(current_data["I_L"])}'
            elif node == 'Mosquitoes':
                size = current_data['I_m'] / 1000 + 1
                label = f'Mosquitoes\nInfected: {int(current_data["I_m"])}'
            else:
                size = 2
                label = 'Environment\n(Rainfall Season)'
            
            circle = Circle((x, y), size, color=node_colors[node], ec='black', lw=2)
            ax.add_patch(circle)
            ax.text(x, y - size - 0.5, label, ha='center', va='top', fontsize=8)
        
        # Draw transmission pathways
        transmission_paths = [
            ('livestock', 'Mosquitoes', 'black'),
            ('Mosquitoes', 'livestock', 'black'),
            ('Mosquitoes', 'Humans', 'red'),
            ('livestock', 'Humans', 'red'),
            ('Environment', 'Mosquitoes', 'blue')
        ]
        
        for start, end, color in transmission_paths:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))

def animate_simulation():
    """Create an animated simulation of the One Health model"""
    # Set up parameters for Kenyan RVF scenario
    params = OneHealthParamters(
        rainfall_factor=1.2,  # Moderate rainfall season
        livestock_vaccination_rate=0.001,  # Low vaccination coverage
        mosquito_control_rate=0.002  # Minimal mosquito control
    )
    
    model = OneHealthSIR(params)
    visualizer = OneHealthVisualizer(model)
    
    # Set up the figure for animation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('One Health SIR Model: Rift Valley Fever in Kenya\n'
                'Human-livestock-Mosquito Dynamics', fontsize=16, fontweight='bold')
    
    # Data storage for animation
    time_data = []
    human_data = {'S': [], 'I': [], 'R': []}
    livestock_data = {'S': [], 'I': [], 'R': []}
    mosquito_data = {'S': [], 'I': []}
    
    def init():
        for ax in axes.flat:
            ax.clear()
        return []
    
    def update(frame):
        # Run one step of the simulation
        if frame > 0:
            model.step()
        
        current_data = model.history[-1]
        time_data.append(current_data['time'])
        
        # Store data for plotting
        human_data['S'].append(current_data['S_h'])
        human_data['I'].append(current_data['I_h'])
        human_data['R'].append(current_data['R_h'])
        
        livestock_data['S'].append(current_data['S_L'])
        livestock_data['I'].append(current_data['I_L'])
        livestock_data['R'].append(current_data['R_L'])
        
        mosquito_data['S'].append(current_data['S_m'])
        mosquito_data['I'].append(current_data['I_m'])
        
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
        
        # Plot 1: Human Population
        axes[0, 0].plot(time_data, human_data['S'], label='Susceptible', color='blue')
        axes[0, 0].plot(time_data, human_data['I'], label='Infected', color='red')
        axes[0, 0].plot(time_data, human_data['R'], label='Recovered', color='green')
        axes[0, 0].set_title('Human Population Dynamics')
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Number of Individuals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: livestock Population
        axes[0, 1].plot(time_data, livestock_data['S'], label='Susceptible', color='blue')
        axes[0, 1].plot(time_data, livestock_data['I'], label='Infected', color='red')
        axes[0, 1].plot(time_data, livestock_data['R'], label='Recovered/Immune', color='green')
        axes[0, 1].set_title('livestock Population Dynamics')
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Number of Animals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mosquito Population
        axes[1, 0].plot(time_data, mosquito_data['S'], label='Susceptible', color='blue')
        axes[1, 0].plot(time_data, mosquito_data['I'], label='Infected', color='red')
        axes[1, 0].set_title('Mosquito Population Dynamics')
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Number of Mosquitoes')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Network Diagram
        visualizer.update_network_diagram(axes[1, 1], current_data)
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=365, init_func=init, 
                        blit=False, repeat=False, interval=100)
    
    plt.tight_layout()
    plt.show()
    
    return anim, model

# Run the simulation and create static plots
def run_comprehensive_analysis():
    """Run a comprehensive analysis with different intervention scenarios"""
    
    print("=== One Health SIR Model: Rift Valley Fever in Kenya ===\n")
    
    # Baseline scenario (no interventions)
    print("1. Baseline Scenario (No Interventions)")
    params_baseline = OneHealthParamters()
    model_baseline = OneHealthSIR(params_baseline)
    results_baseline = model_baseline.run_simulation(365)
    
    # Intervention scenario
    print("2. Intervention Scenario (Vaccination + Mosquito Control)")
    params_intervention = OneHealthParamters(
        livestock_vaccination_rate=0.01,  # 1% vaccination rate
        mosquito_control_rate=0.05,       # 5% mosquito control
        rainfall_factor=0.8               # Dry season
    )
    model_intervention = OneHealthSIR(params_intervention)
    results_intervention = model_intervention.run_simulation(365)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Human cases comparison
    ax1.plot(results_baseline['time'], results_baseline['I_h'], 
            label='Baseline', color='red', linewidth=2)
    ax1.plot(results_intervention['time'], results_intervention['I_h'], 
            label='With Interventions', color='green', linewidth=2)
    ax1.set_title('Human RVF Cases: Baseline vs Interventions')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Number of Infected Humans')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # livestock cases comparison
    ax2.plot(results_baseline['time'], results_baseline['I_L'], 
            label='Baseline', color='red', linewidth=2)
    ax2.plot(results_intervention['time'], results_intervention['I_L'], 
            label='With Interventions', color='green', linewidth=2)
    ax2.set_title('livestock RVF Cases: Baseline vs Interventions')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Number of Infected livestock')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    max_human_baseline = results_baseline['I_h'].max()
    max_human_intervention = results_intervention['I_h'].max()
    reduction = ((max_human_baseline - max_human_intervention) / max_human_baseline) * 100
    
    print(f"\nIntervention Effectiveness:")
    print(f"Peak human cases (baseline): {max_human_baseline:.0f}")
    print(f"Peak human cases (interventions): {max_human_intervention:.0f}")
    print(f"Reduction in peak cases: {reduction:.1f}%")

if __name__ == "__main__":
    # Run the comprehensive analysis
    run_comprehensive_analysis()
    
    # Uncomment the line below to run the animation (may be slow in some environments)
    anim, model = animate_simulation()