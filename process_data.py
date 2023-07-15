import numpy as np
import matplotlib.pyplot as plt
from constants import * 


def main():
    loaded1 = np.loadtxt("a27820320_e0.9_Avar_m100/run0.csv", delimiter=",").T
    loaded2 = np.loadtxt("a27820320_e0.9_Avar_m100/run1.csv", delimiter=",").T
    loaded3 = np.loadtxt("a27820320_e0.9_Avar_m100/run2.csv", delimiter=",").T
    loaded4 = np.loadtxt("a27820320_e0.9_Avar_m100/run3.csv", delimiter=",").T
    loaded5 = np.loadtxt("a27820320_e0.9_Avar_m100/run4.csv", delimiter=",").T
    loaded = [loaded1, loaded2, loaded3, loaded4, loaded5]
    areas = [22500, 15625, 10000, 5625, 2500]
    
    f, a, e, angle, cx, cy, vx, vy, px, py = loaded3
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
    
    index = np.where(v_inf > 115.36146610684378)[0][0]
    print(f"Time: {time[index] / 60 / 60}h")
    print("Distance after 10y:", np.linalg.norm(position[-1]) / AU * 1e3)
    print("Velocity after 10y:", np.linalg.norm(velocity[-1]))
    
    plt.clf()
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].grid()
    ax[1].grid()
    for selected, area in zip(loaded, areas):
        f, a, e, angle, cx, cy, vx, vy, px, py = selected
        time = f
        center = np.vstack((cx, cy)).T
        velocity = np.vstack((vx, vy)).T
        position = np.vstack((px, py)).T
        
        v_norm = np.linalg.norm(velocity, axis=1)
        v_inf = np.sqrt(-MU_SUN/a)
        
        ax[0].plot(time, v_norm, label=f"$A = {area}$ m$^2$")
        ax[1].plot(time, v_inf, label=f"$A = {area}$ m$^2$")
        
    ax[0].set_xlabel("Time [s]")
    ax[1].set_xlabel("Time [s]")
    ax[0].set_xlabel("Velocity [km/s]")
    ax[1].set_ylabel("Velocity [km/s]")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("a27820320_e0.9_Avar_m100/velocity_multiple.pdf", bbox_inches="tight")


if __name__ == "__main__":    
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Helvetica",
        }
    )
    
    main()