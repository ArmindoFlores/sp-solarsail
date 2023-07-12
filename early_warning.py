import matplotlib.pyplot as plt
import numpy as np
from constants import *

LATEX_ENABLED = True
PLOTS = []

def plottable(display=True):
    """Mark a function as plottable. Plottable functions will be drawn in the main() routine.

    Args:
        display (bool, optional): Whether to display this plot. Defaults to True.
    """
    def p(func):
        if display:
            PLOTS.append(func)
        return func
    return p

def radiation_pressure(distance, reflectivness=1):
    """Compute the radiation pressure of the sun on a body.

    Args:
        distance (float): The distance from the body to the sun.
        reflectivness (float, optional): The reflectivness of the body. Defaults to 1.

    Returns:
        float: The applied radiation pressure.
    """
    return AU**2 * ((reflectivness + 1) * G_SC / c) / distance**2

def required_area(distance, mass, reflectivness=1):
    """Compute the required area of a solar sail to stay in a circular orbit of radius
    `distance` with the same period as Earth's.

    Args:
        distance (float): The distance from the sun to the sail.
        mass (float): The sail's mass.
        reflectivness (float, optional): The reflectivness of the sail. Defaults to 1.

    Returns:
        float: The required area to maintain this orbit.
    """
    return mass * (G * M - distance**3 * (2 * np.pi / syear)**2) / radiation_pressure(1, reflectivness)

def cme_warning_time(distance):
    """Compute the minimum and maximum warning time for a solar event with a sail
    at `distance` meters from the Earth.

    Args:
        distance (float): The distance from the Earth to the sail.

    Returns:
        np.ndarray: Returns a pair of the maximum and minimum warning times.
    """
    # https://www.swpc.noaa.gov/phenomena/coronal-mass-ejections
    cme_max_speed = 3.00e6  # m/s
    cme_min_speed = 2.50e5  # m/s
    cme_time = distance / np.array([cme_min_speed, cme_max_speed])
    light_time = distance / c
    return cme_time - light_time    

@plottable(True)
def required_area_reflectivness():
    r = np.linspace(0.5*AU, AU, 5000)
    
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
    for reflectivness in np.linspace(0.5, 1, 5):
        ax.plot(r / AU, required_area(r, 150, reflectivness), label=f"$r={reflectivness}$")  
    
    ax.grid()
    ax.set_xlabel("Distance to the Sun [AU]")
    ax.set_ylabel("Area [$\\mathrm{m}^2$]")
    ax.legend()
    plt.savefig("required_area_reflectivness.pdf", bbox_inches="tight")
    plt.clf()
    
    # How large would it need to be to park at double L1's distance to Earth?
    for distance in (1.5e9, 3e9, 4.5e9):
        print(f"Distance = {distance:.1e} m")
        print(f"\tArea: {required_area(AU - distance, 150, 1):.1f} m^2")
        print(f"\tWarning Time: {cme_warning_time(distance)/60} min")
    
@plottable(False)
def required_area_mass():
    r = np.linspace(0.2*AU, AU, 5000)
    
    for mass in np.linspace(1, 50, 5):
        plt.plot(r / AU, required_area(r, mass, 1), label=f"$m={mass}$")
    
    plt.grid()
    plt.xlabel("Distance to the Sun [AU]")
    plt.ylabel("Area [$\\mathrm{m}^2$]")
    plt.legend()
    plt.savefig("required_area_mass.pdf", bbox_inches="tight")
    plt.clf()

def main():
    for plot_func in PLOTS:
        print(f"Plotting {plot_func.__name__}")
        plot_func()
    

if __name__ == "__main__":
    if LATEX_ENABLED:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.sans-serif": "Helvetica",
            }
        )
    main()