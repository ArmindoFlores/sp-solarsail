import matplotlib.pyplot as plt
import numpy as np
from constants import *

LATEX_ENABLED = True
PLOTS = []

def plottable(display=True):
    def p(func):
        if display:
            PLOTS.append(func)
        return func
    return p

def radiation_pressure(distance, reflectivness=1):
    return AU**2 * ((reflectivness + 1) * G_SC / c) / distance**2

def required_area(distance, mass, reflectivness=1):
    return mass * (G * M - distance**3 * (2 * np.pi / syear)**2) / radiation_pressure(1, reflectivness)

@plottable(True)
def top_speed():
    starting_distance = .5 * AU
    area = 100 * 100
    mass = 150
    
    timeframe = np.linspace(0, 5 * syear, int(1e6))
    r = np.zeros_like(timeframe)
    v = np.zeros_like(timeframe)
    r[0] = starting_distance
    h = timeframe[1] - timeframe[0]
    
    for i in range(1, len(timeframe)):
        a = area / mass * radiation_pressure(r[i-1])
        v[i] = v[i-1] + a * h
        r[i] = r[i-1] + v[i] * h
        
    print(f"top speed: {100 * v[-1] / c} c%")
    print(f"distance: {r[-1] / AU} au")
        
    plt.plot(timeframe, v)
    plt.show()


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