import argparse
import os
import re
import shlex
import threading
import time

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

def output_type(s):
    ss = s.lower()
    if ss in ("hidden", "live", "save"):
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

def simulate(simulator, timestep, accel, count, progress_bar, condition):
    s = time.time()
    for _ in range(count):
        if progress_bar.last_print_n >= 1 and 1000 * (time.time() - s) >= 5:
            time.sleep(1e-3)
            s = time.time()
        with condition:
            simulator.iterate(timestep, accel)
            condition.notify_all()
    progress_bar.write("Simulation complete, continuing display")

def main(args, parser):
    if args.output_type == "save" and not hasattr(args, "frames"):
        print("Please specify the number of frames using '--frames'")
        return
    
    if args.output_type == "save" and not hasattr(args, "output_dir"): 
        print("Please specify the output directory using '--output-dir'")
        return
    
    if hasattr(args, "output_dir") and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Helvetica",
        }
    )
    
    args_list = []
    if hasattr(args, "from_file"):
        with open(args.from_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            custom_args, _ = parser.parse_known_args(shlex.split(line))
            args_list.append(custom_args)
    else:
        args_list.append(args)
    
    if args.output_type == "save":
        fig = plt.figure(figsize=(12, 12))
        if len(args_list) == 1:
            ax = plt.axes(
                xlim=[-1.2*args_list[0].a, 1.2*args_list[0].a] if not hasattr(args, "limits") else args.limits[:2], 
                ylim=[-1.2*args_list[0].a, 1.2*args_list[0].a] if not hasattr(args, "limits") else args.limits[2:]
            )
        else:
            if not hasattr(args, "limits"):
                print("Please specify the bounds using '--limits'")
                return
            ax = plt.axes(
                xlim=args.limits[:2], 
                ylim=args.limits[2:]
            )
    scatters = []
    ellipses = []
    lasts = []
    
    simulation_data = []
    for run_id, custom_args in enumerate(args_list):
        initial_conditions = orbitsim.OrbitalParameters(
            MU_SUN, 
            custom_args.theta, 
            0, 
            a=custom_args.semi_major_axis,
            e=custom_args.eccentricity,
            v=np.array(custom_args.velocity) if hasattr(custom_args, "velocity") else None,
            focus=np.array([0, 0])
        )
        simulator = orbitsim.OrbitSimulator(initial_conditions)
        sim_condition = threading.Condition()
        
        def accel(_, y):
            point = np.array([y[0], y[1]])
            vel = np.array([y[2], y[3]])
            e_r = point - initial_conditions.focus
            r = np.linalg.norm(e_r)
            e_vec = np.cross(vel, np.cross(np.array([*e_r, 0]), vel))[:2] / simulator._mu - e_r / r
            if custom_args.acc_profile == "always" or ((1 if custom_args.acc_profile == "escape_up" else -1) * np.cross(e_vec, e_r) > 0):
                e_r /= r
                solar_radiation_accel = 1e-3 * radiation_pressure(r * 1e3, custom_args.reflectivness) * e_r * custom_args.area / custom_args.mass
                return solar_radiation_accel
            return e_r * 0
        
        progress_bar = tqdm.tqdm(total=args.frames)

        simulator_thread = threading.Thread(target=simulate, args=(simulator, args.timestep, accel, args.frames, progress_bar, sim_condition))
        simulator_thread.start()
        
        if args.output_type != "hidden":
            # Create a figure if needed
            if args.output_type == "live":
                fig = plt.figure(figsize=(6, 6))
                ax = plt.axes(
                    xlim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[:2], 
                    ylim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[2:]
                )
            scatters.append(ax.scatter([], [], s=2, c="#FF0400", linewidths=0))
            lasts.append(ax.scatter([], [], s=8, c="k", linewidths=1))
            ellipses.append(plt.plot([], [], "k--", linewidth=1)[0])
            
            txt = ax.text(.05, .95, "", transform=ax.transAxes, ha="left", va="top") 
            foc = ax.scatter(0, 0, s=24, c="#FFC700", linewidths=0)
            
            def init():
                updated = [txt, foc]
                for last, ellipse, scatter in zip(lasts, ellipses, scatters):
                    last.set_zorder(5)
                    last.set_edgecolor("#000000")
                    updated.append(last)
                    ellipse.set_zorder(4)
                    updated.append(ellipse)
                    scatter.set_zorder(3)
                    updated.append(scatter)
                return updated
            
            def animate(frame):
                updated = [txt, foc]
                if run_id == len(args_list)-1 or args.output_type == "live":
                    progress_bar.update()
                
                with sim_condition:
                    while len(simulator._time) <= frame:
                        sim_condition.wait()
                
                    for i, (last, ellipse, scatter) in enumerate(zip(lasts, ellipses, scatters)):
                        if args.output_type == "live" and i != len(lasts)-1:
                            continue
                        updated += [last, ellipse, scatter]
                        if i == len(lasts)-1 or args.output_type == "live":
                            current_simulator = simulator
                        else:
                            current_simulator = simulation_data[i]
                        scatter.set_offsets(current_simulator._points[max(0, frame-1000):frame+1])
                        last.set_offsets(current_simulator._points[frame])
                        last.set_facecolor("#FFFFFF")
                        
                        if len(args_list) == 1 or args.output_type == "live":
                            if current_simulator._a[frame] > 0:
                                txt.set_text(f"a = {current_simulator._a[frame]:.1f} km\ne = {current_simulator._e[frame]:.4f}\n$\\epsilon$ = {current_simulator._mechanical_energy[frame]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(current_simulator._time[frame])}")
                            else:
                                txt.set_text(f"$v_\infty$ = {np.sqrt(current_simulator._mu / -current_simulator._a[frame]):.2f} km/s\n{time_formatter(current_simulator._time[frame])}")
                            
                        plot_ellipse(ellipse, current_simulator._center[frame], current_simulator._a[frame], current_simulator._b[frame], current_simulator._angle[frame])
                return updated

            plt.gca().set_aspect("equal", adjustable="box")
            
            ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=1, frames=args.frames if hasattr(args, "frames") else None, repeat=False)
            
            if args.output_type == "save" and run_id == len(args_list)-1:
                writervideo = animation.FFMpegWriter(fps=args.framerate, bitrate=12000)
                ani.save(os.path.join(args.output_dir, "trajectory.mp4"), writer=writervideo)
            elif args.output_type == "live":
                plt.show()

            simulator_thread.join()
            progress_bar.close()
            
            if args.output_type == "save" and run_id != len(args_list)-1:
                simulation_data.append(simulator)
            
        else:
            if not hasattr(args, "output_dir") and hasattr(args, "snapshots"):
                print("Missing '--output-dir'")
                return
                
            for frame in range(args.frames):
                progress_bar.update()
                simulator.iterate(args.timestep, accel)
                if hasattr(args, "snapshots") and frame in args.snapshots:
                    plt.clf()
                    fig = plt.figure(figsize=(6, 6))
                    ax = plt.axes(
                        xlim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[:2], 
                        ylim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if args.limits is None else args.limits[2:]
                    )
                    # Initialize a scatter object for the plot
                    scatter1 = ax.scatter([], [], s=2, c="#FF0400", linewidths=0)
                    
                    last = ax.scatter([], [], s=8, c="k", linewidths=1)
                    
                    foc = ax.scatter(0, 0, s=24, c="#FFC700", linewidths=0)
                    ellipse, = plt.plot([], [], "k--", linewidth=1)
                    
                    last.set_zorder(5)
                    ellipse.set_zorder(4)
                    scatter1.set_zorder(3)
                    last.set_edgecolor("#000000")
                    
                    txt = ax.text(.05, .95, "b", transform=ax.transAxes, ha="left", va="top") 
                    scatter1.set_offsets(simulator._points[-1000:-1])
                    last.set_offsets(simulator._points[-1])
                    last.set_facecolor("#FFFFFF")
                    if simulator._a[-1] > 0:
                        txt.set_text(f"a = {simulator._a[-1]:.1f} km\ne = {simulator._e[-1]:.4f}\n$\\epsilon$ = {simulator._mechanical_energy[-1]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(simulator._time[-1])}")
                    else:
                        txt.set_text(f"$v_\infty$ = {np.sqrt(simulator._mu / -simulator._a[-1]):.2f} km/s\n{time_formatter(simulator._time[-1])}")
                    
                    plot_ellipse(ellipse, simulator._center[frame], simulator._a[-1], simulator._b[-1], simulator._angle[-1])
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.savefig(os.path.join(args.output_dir, f"run_{run_id}_frame{frame}.pdf"), bbox_inches="tight")
                    plt.close()
        
        center, v, points = np.array(simulator._center), np.array(simulator._v), np.array(simulator._points)
        np.savetxt(
            os.path.join(args.output_dir, f"run{run_id}.csv"), 
            list(zip(
                simulator._time, 
                simulator._a, 
                simulator._e, 
                simulator._angle, 
                center[:,0], 
                center[:,1], 
                v[:,0], 
                v[:,1], 
                points[:,0],
                points[:,1]
            )),
            delimiter=",", 
        )
    plt.close()
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        prog="interstellar",
        description="Run the orbital simulator for interstellar travel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("area", type=float, default=10000, nargs="?", help="The area of the sail in m\u00b2")
    parser.add_argument("mass", type=float, default=1, nargs="?", help="The mass of the sail in kg")
    parser.add_argument("reflectivness", type=float, default=1, nargs="?", help="The reflectivness of the sail")
    parser.add_argument("-p", "--acc-profile", type=acc_profile, default="always", help="The acceleration profile. There are 3 options: 'always' means the sail is permanently turned to the sun; 'escape_up' and 'escape_down' mean the sail is only turned to the sun in the upper/lower part of the orbit.")
    ic_group = parser.add_argument_group("Initial Conditions", "Initial conditions for the satellite's orbit")
    ic_group.add_argument("-f", "--from-file", type=str, default=argparse.SUPPRESS, help="If specified, load the initial conditions from a file and run a simulation for every orbit presented, overlapping the results")
    ic_group.add_argument("-e", "--eccentricity", type=float, default=0.01671, help="The initial orbit eccentricity")
    ic_group.add_argument("-a", "--semi-major-axis", type=float, default=AU*1e-3, help="The initial orbit's semi-major axis in kilometers")
    ic_group.add_argument("-v", "--velocity", type=as_list(float, 2), default=argparse.SUPPRESS, help="The initial velocity vector in km/s")
    ic_group.add_argument("-t", "--theta", type=float, default=0, help="The initial true anomaly in radians")
    ic_group.add_argument("-T", "--timestep", type=float, default=36000, help="The simulation timestep in seconds")
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("-O", "--output-type", type=output_type, default="live", help="How to output the data. 'live' shows a live plot of the simulation, 'save' generates a video, and 'hidden' doesn't show anything")
    output_group.add_argument("-F", "--frames", type=int, default=argparse.SUPPRESS, help="The total number of frames")
    output_group.add_argument("-r", "--framerate", type=int, default=60, help="The output video frame rate")
    output_group.add_argument("-o", "--output-dir", type=str, default=argparse.SUPPRESS, help="The output files directory")
    output_group.add_argument("-s", "--snapshots", type=as_list(int), default=argparse.SUPPRESS, help="A list of frames to save snapshots of")
    output_group.add_argument("-l", "--limits", type=as_list(float, 4), default=argparse.SUPPRESS, help="The x and y limits of the display window (x_min, x_max, y_min, y_max)")
    
    main(parser.parse_args(), parser)