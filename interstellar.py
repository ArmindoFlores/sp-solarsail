import argparse
import os
import re

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import orbitsim
from constants import *


def as_list(type, minsize=None, maxsize=None):
    def converter(s):
        split = re.split("; ?|, ?| ", s)
        if maxsize is not None and len(split) > maxsize:
            raise ValueError("Too many values") 
        if minsize is not None and len(split) > minsize:
            raise ValueError("Too few values") 
        return tuple(map(type, split))
    return converter

def acc_profile(s):
    ss = s.lower()
    if ss in ("always", "escape_up", "escape_down"):
        return ss
    raise ValueError("Invalid value")

def radiation_pressure(distance, reflectivness=1):
    """Compute the radiation pressure of the sun on a body.

    Args:
        distance (float): The distance from the body to the sun.
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
        args.theta, 
        0, 
        a=args.semi_major_axis,
        e=args.eccentricity,
        v=np.array(args.velocity) if args.velocity is not None else None,
        focus=np.array([0, 0])
    )
    simulator = orbitsim.OrbitSimulator(initial_conditions)
    
    def accel(_, y):
        point = np.array([y[0], y[1]])
        vel = np.array([y[2], y[3]])
        e_r = point - initial_conditions.focus
        r = np.linalg.norm(e_r)
        e_vec = np.cross(vel, np.cross(np.array([*e_r, 0]), vel))[:2] / simulator._mu - e_r / r
        if args.acc_profile == "always" or ((1 if args.acc_profile == "escape_up" else -1) * np.cross(e_vec, e_r) > 0):
            e_r /= r
            solar_radiation_accel = 1e-3 * radiation_pressure(r * 1e3, args.reflectivness) * e_r * args.area / args.mass
            return solar_radiation_accel
        return e_r * 0
    
    if not args.hide_display:
        # Create a figure
        if args.output_dir is not None:
            fig = plt.figure(figsize=(12, 12))
        else:
            fig = plt.figure()
        ax = plt.axes(
            xlim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[:2], 
            ylim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[2:]
        )
        
        # Initialize a scatter object for the plot
        scatter1 = ax.scatter([], [], s=2, c="#FF0400", linewidths=0)
        scatter2 = ax.scatter([], [], s=2, c="#EA2525", linewidths=0)
        scatter3 = ax.scatter([], [], s=2, c="#E85555", linewidths=0)
        
        last = ax.scatter([], [], s=8, c="k", linewidths=1)
        
        foc = ax.scatter(0, 0, s=24, c="#FFC700", linewidths=0)
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
            progress_bar.update(1)      
                
            simulator.iterate(args.timestep, accel)
            
            scatter1.set_offsets(simulator._points[-1000:-1])
            if len(simulator._points) > 1000:
                scatter2.set_offsets(simulator._points[-2000:-1000])
            if len(simulator._points) > 2000:
                scatter3.set_offsets(simulator._points[-3000:-2000])
            last.set_offsets(simulator._points[-1])
            last.set_facecolor("#FFFFFF")
            if simulator._a[-1] > 0:
                txt.set_text(f"a = {simulator._a[-1]:.1f} km\ne = {simulator._e[-1]:.4f}\n$\\epsilon$ = {simulator._mechanical_energy[-1]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(simulator._time[-1])}")
            else:
                txt.set_text(f"$v_\infty$ = {np.sqrt(simulator._mu / -simulator._a[-1]):.2f} km/s\n{time_formatter(simulator._time[-1])}")
            
            plot_ellipse(ellipse, simulator._center, simulator._a[-1], simulator._b[-1], simulator._angle[-1])
            
            return scatter1, scatter2, scatter3, last, foc, txt, ellipse

        # Create the animation
        ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=1, frames=args.frames)
        plt.gca().set_aspect("equal", adjustable="box")
        
        progress_bar = tqdm.tqdm(total=args.frames)
        if args.output_dir is not None:
            writervideo = animation.FFMpegWriter(fps=args.framerate, bitrate=12000)
            ani.save(os.path.join(args.output_dir, "trajectory.mp4"), writer=writervideo)
        else:
            plt.show()
        progress_bar.close()
            
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
        
    else:
        if args.output_dir is None:
            print("Missing '--output-dir'")
            return
            
        for frame in range(args.frames):
            simulator.iterate(args.timestep, accel)
            if args.snapshots and frame in args.snapshots:
                plt.clf()
                fig = plt.figure(figsize=(6, 6))
                ax = plt.axes(
                    xlim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[:2], 
                    ylim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[2:]
                )
                # Initialize a scatter object for the plot
                scatter1 = ax.scatter([], [], s=2, c="#FF0400", linewidths=0)
                scatter2 = ax.scatter([], [], s=2, c="#EA2525", linewidths=0)
                scatter3 = ax.scatter([], [], s=2, c="#E85555", linewidths=0)
                
                last = ax.scatter([], [], s=8, c="k", linewidths=1)
                
                foc = ax.scatter(0, 0, s=24, c="#FFC700", linewidths=0)
                ellipse, = plt.plot([], [], "k--", linewidth=1)
                
                last.set_zorder(5)
                ellipse.set_zorder(4)
                scatter1.set_zorder(3)
                scatter2.set_zorder(2)
                scatter3.set_zorder(1)
                last.set_edgecolor("#000000")
                
                txt = ax.text(.05, .95, "b", transform=ax.transAxes, ha="left", va="top") 
                scatter1.set_offsets(simulator._points[-1000:-1])
                if len(simulator._points) > 1000:
                    scatter2.set_offsets(simulator._points[-2000:-1000])
                if len(simulator._points) > 2000:
                    scatter3.set_offsets(simulator._points[-3000:-2000])
                last.set_offsets(simulator._points[-1])
                last.set_facecolor("#FFFFFF")
                if simulator._a[-1] > 0:
                    txt.set_text(f"a = {simulator._a[-1]:.1f} km\ne = {simulator._e[-1]:.4f}\n$\\epsilon$ = {simulator._mechanical_energy[-1]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(simulator._time[-1])}")
                else:
                    txt.set_text(f"$v_\infty$ = {np.sqrt(simulator._mu / -simulator._a[-1]):.2f} km/s\n{time_formatter(simulator._time[-1])}")
                
                plot_ellipse(ellipse, simulator._center, simulator._a[-1], simulator._b[-1], simulator._angle[-1])
                plt.gca().set_aspect("equal", adjustable="box")
                plt.savefig(os.path.join(args.output_dir, f"frame{frame}.pdf"), bbox_inches="tight")
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        prog="interstellar",
        description="Run the orbital simulator for interstellar travel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("area", type=float, help="The area of the sail in m\u00b2")
    parser.add_argument("mass", type=float, default=1, nargs="?", help="The mass of the sail in kg")
    parser.add_argument("reflectivness", type=float, default=1, nargs="?", help="The reflectivness of the sail")
    parser.add_argument("-p", "--acc-profile", type=acc_profile, default="always", help="The acceleration profile. There are 3 options: 'always' means the sail is permanently turned to the sun; 'escape_up' and 'escape_down' mean the sail is only turned to the sun in the upper/lower part of the orbit.")
    ic_group = parser.add_argument_group("Initial Conditions", "Initial conditions for the satellite's orbit")
    ic_group.add_argument("-e", "--eccentricity", type=float, default=0.01671, help="The initial orbit eccentricity")
    ic_group.add_argument("-a", "--semi-major-axis", type=float, default=AU*1e-3, help="The initial orbit's semi-major axis in kilometers")
    ic_group.add_argument("-v", "--velocity", type=as_list(float, 2), help="The initial velocity vector in km/s")
    ic_group.add_argument("-t", "--theta", type=float, default=0, help="The initial true anomaly in radians")
    ic_group.add_argument("-T", "--timestep", type=float, default=36000, help="The simulation timestep in seconds")
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("-f", "--frames", type=int, help="The total number of frames")
    output_group.add_argument("-r", "--framerate", type=int, default=60, help="The output video frame rate")
    output_group.add_argument("-o", "--output-dir", type=str, help="The output files directory")
    output_group.add_argument("-s", "--snapshots", type=as_list(int), help="A list of frames to save snapshots of")
    output_group.add_argument("-l", "--limits", type=as_list(float, 4), help="The x and y limits of the display window (x_min, x_max, y_min, y_max)")
    output_group.add_argument("--hide-display", action="store_true", default=False, help="Whether to hide the display")
    
    main(parser.parse_args())