import argparse
import importlib
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

def parse_module(string):
	arg_start = string.index("(")
	args_string, mod_string = string[arg_start+1:-1], string[:arg_start]
	module, *path = mod_string.split(".")
	return module, path, args_string

def output_type(s):
    ss = s.lower()
    if ss in ("hidden", "live", "save"):
        return ss
    raise ValueError("Invalid value")

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

def simulate(simulator, timestep, accel, count, progress_bar, condition, args):
    s = time.time()
    for _ in range(count):
        if args.output_type == "hidden":
            progress_bar.update()
        if progress_bar.last_print_n >= 1 and 1000 * (time.time() - s) >= 5:
            time.sleep(1e-3)
            s = time.time()
        with condition:
            simulator.iterate(timestep, accel)
            condition.notify_all()
    if args.output_type != "hidden":
        progress_bar.write("Simulation complete, continuing display")

def main(args, parser):
    if args.output_type == "save" and not hasattr(args, "frames"):
        print("Please specify the number of frames using '--frames'")
        return
    
    if (args.output_type == "save" or (args.output_type == "hidden" and hasattr(args, "snapshots"))) and not hasattr(args, "output_dir"): 
        print("Please specify the output directory using '--output-dir'")
        return
    
    if hasattr(args, "output_dir") and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    figsize = (12, 12) if args.output_type == "video" else (6, 6)
    
    plt.rcParams.update(
        {
            "text.usetex": args.use_latex,
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
    
    if args.output_type in ("save", "hidden") and args.overlap:
        fig = plt.figure(figsize=figsize)
        if len(args_list) == 1:
            ax = plt.axes(
                xlim=[-1.2*args_list[0].semi_major_axis, 1.2*args_list[0].semi_major_axis] if not hasattr(args, "limits") else args.limits[:2], 
                ylim=[-1.2*args_list[0].semi_major_axis, 1.2*args_list[0].semi_major_axis] if not hasattr(args, "limits") else args.limits[2:]
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
            1e-9*G*args.mass, 
            custom_args.theta, 
            0, 
            a=custom_args.semi_major_axis,
            e=custom_args.eccentricity,
            v=np.array(custom_args.velocity) if hasattr(custom_args, "velocity") else None,
            focus=np.array([0, 0])
        )
        simulator = orbitsim.OrbitSimulator(initial_conditions)
        sim_condition = threading.Condition()
        
        if hasattr(custom_args, "acc_profile"):
            module, module_path, module_args = parse_module(custom_args.acc_profile)
            accel_module = importlib.import_module(module)
            accel_func = accel_module
            for path_piece in module_path:
                accel_func = getattr(accel_func, path_piece)
            def accel(t, y):
                return accel_func(t, y, simulator, *eval(module_args))
        else:
            def accel(t, y):
                return np.zeros(2)
        
        progress_bar = tqdm.tqdm(total=args.frames)

        simulator_thread = threading.Thread(target=simulate, args=(simulator, args.timestep, accel, args.frames, progress_bar, sim_condition, args))
        simulator_thread.start()
        
        # Create a figure if needed
        if args.output_type == "live" or not args.overlap:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(
                xlim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if not hasattr(args, "limits") else args.limits[:2], 
                ylim=[-1.2*initial_conditions.a, 1.2*initial_conditions.a] if not hasattr(args, "limits") else args.limits[2:]
            )
        scatters.append(ax.scatter([], [], s=2, c=custom_args.color, linewidths=0))
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
            if run_id == len(args_list)-1 or args.output_type == "live" or not args.overlap:
                progress_bar.update()
            
            with sim_condition:
                while len(simulator._time) <= frame:
                    sim_condition.wait()
            
                for i, (last, ellipse, scatter) in enumerate(zip(lasts, ellipses, scatters)):
                    if (args.output_type == "live" or not args.overlap) and i != len(lasts)-1:
                        continue
                    updated += [last, ellipse, scatter]
                    if i == len(lasts)-1 or args.output_type == "live" or not args.overlap:
                        current_simulator = simulator
                    else:
                        current_simulator = simulation_data[i]
                    scatter.set_offsets(current_simulator._points[max(0, frame-1000):frame+1])
                    last.set_offsets(current_simulator._points[frame])
                    last.set_facecolor("#FFFFFF")
                    
                    if len(args_list) == 1 or args.output_type == "live" or not args.overlap:
                        if current_simulator._a[frame] > 0:
                            txt.set_text(f"a = {current_simulator._a[frame]:.1f} km\ne = {current_simulator._e[frame]:.4f}\n$\\epsilon$ = {current_simulator._mechanical_energy[frame]:.4f} J kg$^\x7b-1\x7d$\n{time_formatter(current_simulator._time[frame])}")
                        else:
                            txt.set_text(f"$v_\infty$ = {np.sqrt(current_simulator._mu / -current_simulator._a[frame]):.2f} km/s\n{time_formatter(current_simulator._time[frame])}")
                        
                    plot_ellipse(ellipse, current_simulator._center[frame], current_simulator._a[frame], current_simulator._b[frame], current_simulator._angle[frame])
            
            if hasattr(args, "snapshots") and frame in args.snapshots:
                plt.savefig(os.path.join(args.output_dir, f"run_{run_id}_frame{frame}.pdf"), bbox_inches="tight")
            
            return updated

        plt.gca().set_aspect("equal", adjustable="box")
        
        ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=1, frames=args.frames if hasattr(args, "frames") else None, repeat=False)
        
        if args.output_type == "save" and (run_id == len(args_list)-1 or not args.overlap):
            writervideo = animation.FFMpegWriter(fps=args.framerate, bitrate=12000)
            ani.save(os.path.join(args.output_dir, f"trajectory_{'full' if args.overlap else run_id}.mp4"), writer=writervideo)
        elif args.output_type == "live":
            plt.show()
        elif args.output_type == "hidden" and (run_id == len(args_list)-1 or not args.overlap):
            if hasattr(args, "snapshots"):
                for frame in args.snapshots:
                    animate(frame)
                    plt.savefig(os.path.join(args.output_dir, f"run_{'full' if args.overlap else run_id}_frame{frame}.pdf"), bbox_inches="tight")

        simulator_thread.join()
        progress_bar.close()
        
        if (args.output_type in ("save", "hidden")) and run_id != len(args_list)-1 and args.overlap:
            simulation_data.append(simulator)
        
        center, v, points = np.array(simulator._center), np.array(simulator._v), np.array(simulator._points)
        if hasattr(args, "output_dir"):
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
                header="time,a,e,angle,center_x,center_y,velocity_x,velocity_y,position_x,position_y"
            )
    plt.close()
        

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(
        prog="interstellar",
        description="Run the orbital simulator for interstellar travel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ic_group = parser.add_argument_group("Initial Conditions", "Initial conditions for the satellite's orbit")
    ic_group.add_argument("-f", "--from-file", type=str, default=argparse.SUPPRESS, help="If specified, load the initial conditions from a file and run a simulation for every orbit presented, overlapping the results")
    ic_group.add_argument("-e", "--eccentricity", type=float, default=0, help="The initial orbit eccentricity")
    ic_group.add_argument("-a", "--semi-major-axis", type=float, default=AU*1e-3, help="The initial orbit's semi-major axis in kilometers")
    ic_group.add_argument("-v", "--velocity", type=as_list(float, 2), default=argparse.SUPPRESS, help="The initial velocity vector in km/s")
    ic_group.add_argument("-t", "--theta", type=float, default=0, help="The initial true anomaly in radians")
    ic_group.add_argument("-m", "--mass", type=float, default=M, help="The mass of the center body in kilograms")
    ic_group.add_argument("-T", "--timestep", type=float, default=36000, help="The simulation timestep in seconds")
    ic_group.add_argument("-p", "--acc-profile", type=str, default=argparse.SUPPRESS, help="The acceleration profile. Should have the format of modulename.function(param1, param2, ...). 'modulename' will be imported and 'function' will be called with the parameters (t, y, simulator, param1, param2, ...) where t is the simulation time and y is a vector with (px, py, vx, vy).")
    ic_group.add_argument("-C", "--color", type=str, default="r", help="The color to plot the trajectory as. Accepts any format matplotlib accepts.")
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("-O", "--output-type", type=output_type, default="live", help="How to output the data. 'live' shows a live plot of the simulation, 'save' generates a video, and 'hidden' doesn't show anything")
    output_group.add_argument("-F", "--frames", type=int, default=argparse.SUPPRESS, help="The total number of frames")
    output_group.add_argument("-r", "--framerate", type=int, default=60, help="The output video frame rate")
    output_group.add_argument("-o", "--output-dir", type=str, default=argparse.SUPPRESS, help="The output files directory")
    output_group.add_argument("-s", "--snapshots", type=as_list(int), default=argparse.SUPPRESS, help="A list of frames to save snapshots of")
    output_group.add_argument("-l", "--limits", type=as_list(float, 4), default=argparse.SUPPRESS, help="The x and y limits of the display window (x_min, x_max, y_min, y_max)")
    output_group.add_argument("-E", "--overlap", action="store_true", default=False, help="Whether multiple trajectories should overlap")
    output_group.add_argument("-L", "--use-latex", action="store_true", default=False, help="Whether to use LaTeX for rendering. WARNING: This takes a long time")
    
    main(parser.parse_args(), parser)