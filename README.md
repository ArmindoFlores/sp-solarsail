# Space Physics Project (2022/2023)
This repository holds the scripts used to generate the data used in the final report for the Space Physics course at IST.

## Files
- `constants.py`: defines physical constants used throughout the project
- `early_warning.py`: contains the code related to the calculations pertaining to the Early Warning System discussed in the project
- `interstellar.py`: a script to run the orbital simulator
- `orbitsim.py`: contains an orbital simulator

## Orbit Simulator
The orbit simulator can be used via the utility program `interstellar.py`. All the possible command line arguments can be consulted using `python interstellar.py -h`.

## Usage
This script will simulate an orbit with the specified initial conditions. To specify an initial condition, use one of the associated arguments:

* Semi-major axis (kilometers)- `--semi-major-axis`/`-a` 
* Eccentricity - `--eccentricity`/`-e` 
* True anomaly (radians) - `--theta`/`-t` 
* Velocity (kilometers per second) - `--velocity`/`-v` 
* Focus coordinates (kilometers) - `--focus-coordinates`

For example, the following command will simulate the Earth's orbit around the Sun: `python interstellar.py -a 1.495979e+8 -e 0.01671 --frames 1000`. The last argument, `--frames`, specifies the number of iterations to simulate the system for.

These parameters can also be loaded from a file and follow the same syntax. For this, the `--from-file` option is used. All initial conditions not specified in the file are ignored. An example file would be:

```
-a 1.495979e+8 -e 0.01671
-a 2.279872e+08 -e 0.0934
```

and using `python interstellar.py --from-file file.txt --frames 1000 --limits -3e+08,3e+08,-3e+08,3e+08` we would now be able to see both the orbit of the Earth and of Mars. The `--limits` flag is now required for the bounds of the drawn region.

To introduce propulsion, a new python script must be created This script must define a function that takes in at least three parameters, `t`, `y`, and `orbital_params`. `t` is the simulation time at call time, `y` is a vector of length four containing `[pos_x, pos_y, vel_x, vel_y]`, and `orbital_params` is a custom object that contains information about the simulator. For example, consider the file `acceleration_profiles.py`:

```python
import numpy as np

def tangential(t, y, orbital_params, mag=0.000007):
    point = np.array([y[0], y[1]])
    e_r = point - orbital_params.focus
    e_t = np.array([-e_r[1], e_r[0]])
    e_t /= np.linalg.norm(e_t)
    return e_t * mag
```

The `tangential(...)` function returns a vector tangential to the trajectory at the current point and with a magnitude of `7e-6`. We could also change the function signature to include additional parameters like the mass of the spacecraft, for example. 

To use this function with the simulator, add the `--acc-profile MODULENAME.PATH.TO.FUNCTION(AARG1, AARG2, ...)`, replacing MODULENAME with the name of the created file and the path to the function with its path. `AARG[N]` are custom arguments, like the `mag` argument in the `tangential(...)` function, which must be specified here. Here is an example using the previously defined profile: `python interstellar.py -a 1.495979e+8 -e 0.01671 --frames 1000 --acc-profile "acceleration_profiles.tangential(5e-6)"`.

There is also a list of other arguments that deal with the output of this script:

* `-o`/`--output-dir` - Used to specify a directory where all generated files will end up.
* `-O`/`--output-type` - Can be either `live`, `save`, or `hidden`. When set to `live` (the default), a window will open and show the simulation in real time. `save`, on the other hand, generates a video output. `hidden` does neither.
* `-F`/`--frames` - As previously discussed, this is used to specify the total number of frames to be drawn.
* `-r`/`--framerate` - Used to set the video framerate when using `-O save`.
* `-s`/`--snapshots` - A list of frames to save as images. For example, `-s 1,5,10` will save the frames 1, 5, and 10 to a file.
* `-l`/`--limits` - The `x` and `y` limits of the display window, in the form of `--limits xmin,xmax,ymin,ymax`. 

## Examples
### Example 1 - Spacecraft escaping from the solar system

**Input:** `python interstellar.py -a 27820320 -e 0.9 --theta 4.084070449666731 --timestep 100 -O live -o example_1 --frames 900 --limits "-1.3e7 0.3e7 -1e7 0.6e7" --acc-profile="acceleration_profiles.escape_up(1.0, 10000, 75)" --use-latex`

<p align="center">
    <img src="/examples/example_1.gif" width="400">
</p>

**Legend:** In dashed and dash-dotted lines, the current elliptical and hyperbolic orbits, respectively. The trajectory from the beginning of the times is represented as a blue solid line.

### Example 2 - Comparison between 3 different trajectories

**Input:** `python interstellar.py --from-file comparison.txt --frames 700 --limits "-0.75e7 0.75e7 -0.4e7 1.1e7" --timestep 100 -O live -o example_2 -E --use-latex`

<p align="center">
    <img src="/examples/example_2.gif" width="400">
</p>

**Legend:** The yellow one is an elliptical trajectory where $\alpha = \frac{\pi}{2}$ and thus no pressure is felt; the blue one is our chosen trajectory for the maneuver and we can see that after just 13 hours and 53 minutes, it is already on an escape course from the solar system; the orange trajectory has the sail is tilted orthogonally to the radiation source ($\alpha = 0$) since the beginning (before $\theta = 0$, the perihelion) and we can see it has not yet managed to escape the Sun's gravity well.

### Example 3 - Trajectories of sails with different areas

**Input:** `python interstellar.py --from-file escape.txt --timestep 100 -O live -o example_3 --frames 500 --limits "-1.3e7 0.3e7 -1e7 0.6e7" -E --use-latex`

<p align="center">
    <img src="/examples/example_3.gif" width="400">
</p>

**Legend:** Sail 1 - $22500 \ \mathrm{m}^2$; Sail 2 - $15625 \ \mathrm{m}^2$; Sail 3 - $10000 \ \mathrm{m}^2$; Sail 4 - $5625 \ \mathrm{m}^2$; Sail 5 - $2500 \ \mathrm{m}^2$.

### Example 4 - Early Warning System

**Input:** `python interstellar.py -a 74798950 --velocity 0,14.892  -e 0 --frames 789 --timestep 40000 -O live -o example_4 --limits "-1.5e8 1.5e8 -1e8 2e8" --acc-profile="acceleration_profiles.always(1, 85807.455, 150)" --use-latex`

<p align="center">
    <img src="/examples/example_4.gif" width="400">
</p>

**Legend:** Simulated orbit of an early warning system satellite at different times.

## Running on Mac
If a warning appears within the progress bar saying `ApplePersistenceIgnoreState: Existing state will not be touched...`, the command `defaults write org.python.python ApplePersistenceIgnoreState NO` ran on the terminal should fix it.