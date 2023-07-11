import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

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

def time_formatter(time):
    """Formats a time value

    Args:
        time (float): time in seconds.
        
    Returns:
        str: The formatted string.
    """
    if time < 60:
        return f"{time:.1f}s"
    if time < 3600:
        minutes = int(time)//60
        seconds = time - minutes*60
        return f"{minutes:02d}m {seconds:.1f}s"
    if time < 86400:
        hours = int(time)//3600
        time -= hours * 3600
        minutes = int(time)//60
        time -= minutes * 60
        seconds = time
        return f"{hours:02d}h {minutes:02d}m {seconds:.1f}s"
    if time < 31558464:
        days = int(time)//86400
        time -= days * 86400
        hours = int(time)//3600
        time -= hours * 3600
        minutes = int(time)//60
        time -= minutes * 60
        seconds = time
        return f"{days:03d}d {hours:02d}h {minutes:02d}m {seconds:.1f}s"
    years = int(time)//31558464
    time -= years * 31558464
    days = int(time)//86400
    time -= days * 86400
    hours = int(time)//3600
    time -= hours * 3600
    minutes = int(time)//60
    time -= minutes * 60
    seconds = time
    return f"{years}y {days:03d}d {hours:02d}h {minutes:02d}m {seconds:.1f}s"

def plot_ellipse(line, center, a, b, angle, n_points=100, angle_start=0, angle_finish=2*np.pi):
    theta = np.linspace(angle_start, angle_finish, n_points)
    x = center[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y = center[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    return line.set_data(x, y)

def main(args):
    if args.output_dir is not None and args.frames is None:
        print("Please specify the number of frames using '--frames'")
        return
    
    if args.output_dir is not None and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
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
        vel = np.array([y[2], y[3]])
        e_r = point - initial_conditions.focus
        r = np.linalg.norm(e_r)
        e_vec = np.cross(vel, np.cross(np.array([*e_r, 0]), vel))[:2] / simulator._mu - e_r / r
        # e = np.linalg.norm(e_vec)
        # a = -simulator._mu / (2 * mech_e)
        if np.cross(e_vec, e_r) > 0:
            e_r /= r
            solar_radiation_accel = radiation_pressure(r * 1e3) * e_r * args.area * 1e-6 / args.mass
            return solar_radiation_accel
        return e_r * 0
    
    # Create a figure
    if args.output_dir is not None:
        fig = plt.figure(figsize=(12, 12))
    else:
        fig = plt.figure()
    ax = plt.axes(
        xlim=[-4*initial_conditions.a, 4*initial_conditions.a], 
        ylim=[-4*initial_conditions.a, 4*initial_conditions.a]
    )
    
    # Initialize a scatter object for the plot
    scatter1 = ax.scatter([], [], s=2, c="#FF0400", linewidths=0)
    scatter2 = ax.scatter([], [], s=2, c="#EA2525", linewidths=0)
    scatter3 = ax.scatter([], [], s=2, c="#E85555", linewidths=0)
    
    last = ax.scatter([], [], s=8, c="k", linewidths=1)
    
    foc = ax.scatter(0, 0, s=8, c="b", linewidths=0)
    ellipse, = plt.plot([], [], "k--", linewidth=1)
    
    txt = ax.text(.05, .95, "b", transform=ax.transAxes, ha="left", va="top") 
    
    def init():
        last.set_zorder(5)
        ellipse.set_zorder(4)
        scatter1.set_zorder(3)
        scatter2.set_zorder(2)
        scatter3.set_zorder(1)
        last.set_edgecolor("#000000")
        return scatter1, scatter2, scatter3, last, foc, txt, ellipse
    
    # Define the animation function
    def animate(_):
        if args.output_dir is not None:
            progress_bar.update(1)      
            
        simulator.iterate(args.timestep, accel)
        
        scatter1.set_offsets(simulator._points[-1000:-1])
        if len(simulator._points) > 1000:
            scatter2.set_offsets(simulator._points[-2000:-1000])
        if len(simulator._points) > 2000:
            scatter3.set_offsets(simulator._points[-3000:-2000])
        last.set_offsets(simulator._points[-1])
        if simulator._points[-1][0] < 0:
            last.set_facecolor("#FFFFFF")
        else:
            last.set_facecolor("#FF0000")
        if simulator._a[-1] > 0:
            txt.set_text(f"a = {simulator._a[-1]:.1f} km\ne = {simulator._e[-1]:.4f}\n$\\epsilon$ = {simulator._mechanical_energy[-1]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(simulator._time[-1])}")
        else:
            txt.set_text(f"$v_\infty$ = {np.sqrt(simulator._mu / -simulator._a[-1]):.2f} km/s\n{time_formatter(simulator._time[-1])}")
        
        plot_ellipse(ellipse, simulator._center, simulator._a[-1], simulator._b[-1], simulator._angle[-1])
        
        return scatter1, scatter2, scatter3, last, foc, txt, ellipse

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=1, frames=args.frames)
    plt.gca().set_aspect("equal", adjustable="box")
    
    if args.output_dir is not None:
        writervideo = animation.FFMpegWriter(fps=args.framerate, bitrate=12000)
        progress_bar = tqdm.tqdm(total=args.frames)
        ani.save(os.path.join(args.output_dir, "trajectory.mp4"), writer=writervideo)
        progress_bar.close()
    else:
        plt.show()
        
    plt.clf()
    plt.grid()
    plt.ylabel("Mechanical Energy, $\\epsilon$ [J kg $^{-1}$]")
    plt.xlabel("Time [s]")
    plt.plot(simulator._time, simulator._mechanical_energy)
    
    if args.output_dir is not None:
        plt.savefig(os.path.join(args.output_dir, "mechanical_energy.pdf"), bbox_inches="tight")
    else:
        plt.show()
        
    plt.clf()
    plt.grid()
    plt.ylabel("Eccentricity")
    plt.xlabel("Time [s]")
    plt.plot(simulator._time, simulator._e)
    
    if args.output_dir is not None:
        plt.savefig(os.path.join(args.output_dir, "eccentricity.pdf"), bbox_inches="tight")
    else:
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
    parser.add_argument("-r", "--framerate", type=int, default=60, help="The output video frame rate")
    parser.add_argument("-o", "--output-dir", type=str, help="The output files directory")
    
    main(parser.parse_args())