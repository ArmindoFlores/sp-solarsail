import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import orbitsim
from constants import *


def radiation_pressure(distance, reflectivness=1):
    """Compute the radiation pressure of the sun on a body.

    Args:
        distance (flaot): The distance from the body to the sun.
        reflectivness (float, optional): The reflectivness of the body. Defaults to 1.

    Returns:
        float: The applied radiation pressure.
    """
    return AU**2 * ((reflectivness + 1) * G_SC / c) / distance**2

def main(args):
    if args.output_file is not None and args.frames is None:
        print("Please specify the number of frames using '--frames'")
        return
    
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Helvetica",
        }
    )
    
    initial_conditions = orbitsim.OrbitalParameters(
        MU_SUN, 
        0, 
        0, 
        a=args.semi_major_axis,
        e=args.eccentricity,
        focus=np.array([0, 0])
    )
    simulator = orbitsim.OrbitSimulator(initial_conditions)
    
    def accel(_, y):
        point = np.array([y[0], y[1]])
        e_r = point - initial_conditions.focus
        if e_r[0] > 0:
            r = np.linalg.norm(e_r)
            e_r /= r
            solar_radiation_accel = radiation_pressure(r * 1e3) * e_r * args.area * 1e-6 / args.mass
            return solar_radiation_accel
        return e_r * 0
    
    # Create a figure
    fig = plt.figure()
    ax = plt.axes(
        xlim=[-1.5*initial_conditions.a, 1.5*initial_conditions.a], 
        ylim=[-1.5*initial_conditions.a, 1.5*initial_conditions.a]
    )
    
    # Initialize a scatter object for the plot
    scatter3 = ax.scatter([], [], s=1, c="#E85555")
    scatter2 = ax.scatter([], [], s=1, c="#EA2525")
    scatter1 = ax.scatter([], [], s=1, c="#FF0400")
    last = ax.scatter([], [], s=2, c="k")
    foc = ax.scatter(0, 0, s=4, c="b")
    txt = ax.text(.05, .95, "b", transform=ax.transAxes, ha="left", va="top") 
    
    def init():
        return scatter1, scatter2, scatter3, last, foc, txt
    
    # Define the animation function
    def animate(_):
        simulator.iterate(args.timestep, accel)
        scatter1.set_offsets(simulator._points[-1000:-1])
        if len(simulator._points) > 1000:
            scatter2.set_offsets(simulator._points[-2000:-1000])
        if len(simulator._points) > 2000:
            scatter3.set_offsets(simulator._points[-3000:-2000])
        last.set_offsets(simulator._points[-1])
        if simulator._points[-1][0] < 0:
            last.set_color("#000000")
        else:
            last.set_color("#FF6A00")
        txt.set_text(f"a = {simulator._a[-1]:.1f} km\ne = {simulator._e[-1]:.4f}\n$\\epsilon$ = {simulator._mechanical_energy[-1]:.4f} J kg$^\x7b-1\x7d$\n")
        return scatter1, scatter2, scatter3, last, foc, txt

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=1, frames=args.frames)
    plt.gca().set_aspect("equal", adjustable="box")
    
    if args.output_file is not None:
        writervideo = animation.FFMpegWriter(fps=60, bitrate=12000)
        ani.save(args.output_file, writer=writervideo)
    else:
        plt.show()
        plt.grid()
        plt.plot(simulator._time, simulator._mechanical_energy)
        plt.show()
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        prog="interstellar",
        description="Run the orbital simulator for interstellar travel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("area", type=float, help="The area of the sail in m^2")
    parser.add_argument("mass", type=float, help="The mass of the sail in kg")
    parser.add_argument("-e", "--eccentricity", type=float, default=0.01671, help="The initial orbit eccentricity")
    parser.add_argument("-a", "--semi-major-axis", type=float, default=AU*1e-3, help="The initial orbit's semi-major axis in kilometers")
    parser.add_argument("-t", "--timestep", type=float, default=36000, help="The simulation timestep in seconds")
    parser.add_argument("-f", "--frames", type=int, help="The total number of frames")
    parser.add_argument("-o", "--output-file", type=str, help="The output movie file")
    
    main(parser.parse_args())