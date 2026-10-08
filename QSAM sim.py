"""
QSAM Acoustic Field Simulator
-----------------------------
Simulates the acoustic pressure field of an ultrasonic phased array
and visualizes the intersection of optical targeting beams.

Algorithm: Superposition of sinusoidal point sources.
Author: [Your Name/Pseudonym]
Date: October 2026
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- Constants & Configuration ---
FREQ = 40000.0          # Frequency in Hz
SPEED_OF_SOUND = 343.0  # Velocity in air at 20°C (m/s)
NUM_EMITTERS = 16       # Linear array count
PITCH = 0.01            # Transducer center-to-center spacing (m)
RESOLUTION = 300        # Grid density (pixels per axis)

def calculate_wavenumber(freq: float, speed: float) -> float:
    """Returns the angular wavenumber k."""
    wavelength = speed / freq
    return 2 * np.pi / wavelength

def generate_pressure_field(
    target_pos: Tuple[float, float], 
    grid_res: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the complex pressure field for a phased array focused at target_pos.
    
    Args:
        target_pos: (x, y) coordinates of the focal point in meters.
        grid_res: Resolution of the simulation mesh.
        
    Returns:
        X, Y (meshgrids), and Intensity (normalized pressure magnitude).
    """
    # 1. Setup Physical Space (20cm x 20cm)
    x_range = np.linspace(-0.1, 0.1, grid_res)
    y_range = np.linspace(0, 0.2, grid_res)
    X, Y = np.meshgrid(x_range, y_range)

    # 2. Setup Emitter Array Geometry
    # Center the array at y=0, spread along x-axis
    width = (NUM_EMITTERS - 1) * PITCH
    emitter_x = np.linspace(-width / 2, width / 2, NUM_EMITTERS)
    emitter_y = np.zeros(NUM_EMITTERS)

    # 3. Beam Steering Logic (Phased Array)
    k = calculate_wavenumber(FREQ, SPEED_OF_SOUND)
    phases = []
    
    # Calculate required phase delay for constructive interference at target
    for ex in emitter_x:
        dist_to_target = np.sqrt((target_pos[0] - ex)**2 + (target_pos[1] - 0)**2)
        # Phase delay = -k * distance (rewinding the wave to arrive synchronously)
        phases.append(k * dist_to_target)

    # 4. Superposition Loop
    total_pressure = np.zeros_like(X, dtype=complex)
    
    for i in range(NUM_EMITTERS):
        # Distance map from this specific emitter to every pixel
        dist_grid = np.sqrt((X - emitter_x[i])**2 + (Y - emitter_y[i])**2)
        
        # Wave function: A * e^(i(kd - phase))
        # Note: Amplitude decay (1/r) is omitted for near-field visualization clarity
        wave = np.exp(1j * (k * dist_grid - phases[i]))
        total_pressure += wave

    return X, Y, np.abs(total_pressure)

def plot_qsam_field(target: Tuple[float, float]):
    """Renders the acoustic matrix and optical intersection."""
    X, Y, intensity = generate_pressure_field(target, RESOLUTION)

    plt.figure(figsize=(10, 8), dpi=120)
    
    # 1. Plot Acoustic Field (The Matrix)
    plt.pcolormesh(X, Y, intensity, cmap='inferno', shading='auto')
    plt.colorbar(label='Acoustic Pressure (Normalized)')

    # 2. Visualize Emitters
    array_width = (NUM_EMITTERS - 1) * PITCH
    plt.scatter(
        np.linspace(-array_width/2, array_width/2, NUM_EMITTERS), 
        np.zeros(NUM_EMITTERS), 
        color='cyan', s=20, label='Ultrasonic Array'
    )

    # 3. Visualize Quantum Excitation (Lasers)
    # Draw laser paths intersecting at the target
    plt.plot(
        [-0.1, target[0]], [target[1], target[1]], 
        color='lime', linestyle='--', linewidth=1, alpha=0.7, label='Laser Beam A (IR)'
    )
    plt.plot(
        [target[0], target[0]], [0.2, target[1]], 
        color='lime', linestyle='--', linewidth=1, alpha=0.7, label='Laser Beam B (IR)'
    )
    
    # 4. Highlight the Voxel
    plt.scatter(
        target[0], target[1], 
        color='white', edgecolors='lime', s=100, linewidth=2, zorder=10, 
        label='QSAM Voxel (Excited)'
    )

    # Formatting
    plt.title(f'QSAM Simulation: Node Formation at {target}m')
    plt.xlabel('Lateral Position (m)')
    plt.ylabel('Axial Depth (m)')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(False)
    plt.axis('equal')
    
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    # Define target voxel coordinates (x, y) in meters
    VOXEL_TARGET = (0.0, 0.10) 
    plot_qsam_field(VOXEL_TARGET)
