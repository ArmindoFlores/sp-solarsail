import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from constants import *


def fitfunc(x, a, b, c):
    return c + a * np.sqrt(x - b)

def process_interstellar():
    files = list(filter(lambda file: file.startswith("run") and file.endswith(".csv"), os.listdir("a27820320_e0.9_Avar_m100")))
    loaded = [np.loadtxt(os.path.join("a27820320_e0.9_Avar_m100", file), delimiter=",").T for file in files][:-1]
    areas = [22500.0, 19062.5, 15625.0, 12812.5, 10000.0, 7812.5, 5625.0, 2500.0][:-1]
    
    f, a, e, angle, cx, cy, vx, vy, px, py = loaded[4]
    time = f
    center = np.vstack((cx, cy)).T
    velocity = np.vstack((vx, vy)).T
    position = np.vstack((px, py)).T
    
    v_inf = np.sqrt(-MU_SUN/a)
    v_norm = np.linalg.norm(velocity, axis=1)
    
    _, ax = plt.subplots(nrows=1, ncols=1)
    ax.grid()
    ax.plot(time, np.linalg.norm(velocity, axis=1), label="$|v|$")
    ax.plot(time, v_inf, label="$v_\infty$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [km/s]")
    plt.legend()
    plt.savefig("a27820320_e0.9_Avar_m100/velocity_single.pdf", bbox_inches="tight")
    plt.clf()
    
    # _, ax = plt.subplots(nrows=1, ncols=1)
    # ax.grid()
    # ax.plot(time, -MU_SUN/(2 * a), label="$|v|$")
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Energy [J/kg]")
    # plt.legend()
    # plt.show()
    # plt.clf()
    
    # index = np.where(v_inf > 115.36146610684378)[0][0]
    # print(f"Time: {time[index] / 60 / 60}h")
    # print("Distance after 10y:", np.linalg.norm(position[-1]) / AU * 1e3)
    # print("Velocity after 10y:", np.linalg.norm(velocity[-1]))
    
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].grid()
    ax[1].grid()
    final_velocities = []
    for selected, area in zip(loaded, areas):
        f, a, e, angle, cx, cy, vx, vy, px, py = selected
        time = f
        center = np.vstack((cx, cy)).T
        velocity = np.vstack((vx, vy)).T
        position = np.vstack((px, py)).T
        
        v_norm = np.linalg.norm(velocity, axis=1)
        v_inf = np.sqrt(-MU_SUN/a)
        # final_velocities.append(np.linalg.norm(velocity[-1]))
        if a[-1] < 0:
            final_velocities.append((area, np.sqrt(-MU_SUN/a[-1])))
        
        ax[0].plot(time, v_norm, label=f"$A = {area}$ m$^2$")
        ax[1].plot(time, v_inf, label=f"$A = {area}$ m$^2$")
    final_velocities = np.array(final_velocities)
    final_velocities_x = final_velocities.T[0]
    final_velocities_y = final_velocities.T[1]
        
    ax[0].set_xlabel("Time [s]")
    ax[1].set_xlabel("Time [s]")
    ax[0].set_xlabel("Velocity [km/s]")
    ax[1].set_ylabel("Velocity [km/s]")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("a27820320_e0.9_Avar_m100/velocity_multiple.pdf", bbox_inches="tight")
    
    plt.clf()
    plt.close()
    
    result = curve_fit(fitfunc, final_velocities_x, final_velocities_y, (1, 0, 0))[0]
    fit = lambda x: fitfunc(x, *result)
    
    print(f"fit(x) = {result[0]:.3e} * sqrt(x - {result[1]:.3e}) + {result[2]:.3e}")
    
    MSE = np.mean(np.square(fit(final_velocities_x) - final_velocities_y))
    print(f"{MSE = }")
    
    # plt.ylim([0, 1.05*max(final_velocities_y)])
    # plt.xlim([0, 1.1*max(final_velocities_x)])
    for area, vel in final_velocities:
        # line1, = plt.plot([area, area], [0, vel], "k--", linewidth=1)
        # line2, = plt.plot([0, area], [vel, vel], "k--", linewidth=1)
        # line1.set_zorder(1)
        # line2.set_zorder(1)
        point, = plt.plot([area], [vel], "ro", markersize=5)
        point.set_zorder(3)
        
    x1 = np.linspace(min(final_velocities_x), max(final_velocities_x), 75)
    x2 = np.linspace(0, min(final_velocities_x), 15)
    x3 = np.linspace(max(final_velocities_x), 1.1*max(final_velocities_x), 10)
    graph1, = plt.plot(x1, fit(x1), "k", linewidth=1)
    graph2, = plt.plot(x2, fit(x2), "k--", linewidth=1)
    graph3, = plt.plot(x3, fit(x3), "k--", linewidth=1)
    graph1.set_zorder(2)
    graph2.set_zorder(2)
    graph3.set_zorder(2)
    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.xlabel("Area [m$^2$]")
    plt.ylabel("$v_\infty$ [km s$^{-1}$]")
    plt.savefig("a27820320_e0.9_Avar_m100/v_inf_area.pdf", bbox_inches="tight")
    plt.show()

def process_earth():
    loaded = np.loadtxt("earth_escape/run0.csv", delimiter=",").T
    f, a, e, angle, cx, cy, vx, vy, px, py = loaded
    
    plt.plot(f / syear, a)
    plt.show()
    

if __name__ == "__main__":    
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Helvetica",
        }
    )
    
    process_interstellar()
    # process_earth()