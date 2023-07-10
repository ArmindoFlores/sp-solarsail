import numpy as np
import scipy.integrate


def rotmat2d(angle):
    """Create a 2D rotation matrix by `angle` radians.

    Args:
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: A 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def tangencial_vector(theta, a, b, angle=0):
    """Compute the vector tangential to an ellipse at an angle of `theta`.

    Args:
        theta (float): The true anomaly, in radians.
        a (float): The ellipse's semi-major axis.
        b (float): The ellipse's semi-minor axis.
        angle (float, optional): The angle of the ellipse's axis to the x axis. Defaults to 0.

    Returns:
        np.ndarray: A vector of length 2 tangential to the ellipse.
    """
    e = np.sqrt(1 - (b/a)**2)
    E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))
    if theta > np.pi:
        E = -E
    return rotmat2d(angle) @ np.array([
        -a * np.sin(E),
        b * np.cos(E)
    ]) / np.sqrt(b**2 * np.cos(E)**2 + a**2 * np.sin(E)**2)
    
def orbit_equation(theta, a, e):
    """Computes `r`, the distance to the central body.

    Args:
        theta (float): The true anomaly.
        a (float): The orbit's semi-major axis.
        e (float): The orbit's eccentricity.

    Returns:
        float: The distance to the central body.
    """
    return a * (1 - e**2) / (1 + e * np.cos(theta))

def compute_angle(vector1, vector2):
    """Compute the angle between two vectors in radians.

    Args:
        vector1 (np.ndarray): The first vector.
        vector2 (np.ndarray): The second vector.

    Returns:
        float: The angle between `vector1` and `vector2`.
    """
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_angle = dot_product / norm_product
    angle = np.arccos(cosine_angle)
    return angle

def on_ellipse(theta, focus, a, b):
    """Compute the (x, y) coordinates of a point on an ellipse

    Args:
        theta (float): The true anomaly
        focus (np.ndarray): The Cartesian coordinates of the ellipse's focus
        a (float): The ellipse's semi-major axis
        b (float): The ellipse's semi-minor axis

    Returns:
        np.ndarray: The (x, y) coordinates of the point
    """
    r = orbit_equation(theta, a, np.sqrt(1 - (b/a)**2))
    return focus + r * np.array([np.cos(theta), np.sin(theta)])

def flight_path_angle(v, position, focus):
    """Compute the flight path angle at a point on an orbit

    Args:
        v (np.ndarray): The current velocity using Cartesian coordinates
        position (np.ndarray): The current position
        focus (np.ndarray): The coordinates of the center body

    Returns:
        float: The flight path angle
    """
    r = position - focus
    return np.pi/2 - np.arctan2(np.cross(r, v), np.dot(r, v))

def velocity_at(theta, a, b, mu, angle=0, vector=True):
    """Compute the orbital velocity at a point in the orbit

    Args:
        theta (float): The true anomaly
        a (float): The orbit's semi-major axis
        b (float): The orbit's semi-minor axis
        mu (float): mu is the product of the gravitational constant G with
        the mass of the center body M
        angle (float, optional): The angle the orbit's ellipse makes with the x axis. Defaults to 0.
        vector (bool, optional): Whether to return a vector. If false, returns the magnitude. Defaults to True.

    Returns:
        np.ndarray | float: The velocity (either a vector or its magnitude)
    """

    v = np.sqrt(mu * (2 / orbit_equation(theta, a, np.sqrt(1 - (b/a)**2)) - 1 / a))
    if vector:
        return tangencial_vector(theta, a, b, angle) * v
    return v

def runge_kutta_step(f, x, t, h):
    k1 = h * f(t, x)
    k2 = h * f(t + 0.5*h, x + 0.5*k1)
    k3 = h * f(t + 0.5*h, x + 0.5*k2)
    k4 = h * f(t + h, x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6


class OrbitalParameters:
    A_REQUIREMENTS = (
        (("b", "c"), lambda b, c: np.sqrt(b**2 + c**2)),
        (("b", "e"), lambda b, e: np.sqrt(b**2 / (1 - e**2))),
        (("e", "c"), lambda e, c: c / e),
    )
    B_REQUIREMENTS = (
        (("a", "c"), lambda a, c: np.sqrt(a**2 - c**2)),
        (("a", "e"), lambda a, e: np.sqrt(a**2 * (1 - e**2))),
        (("e", "c"), lambda e, c: np.sqrt((c / e)**2 - c**2))
    )
    C_REQUIREMENTS = (
        (("a", "b"), lambda a, b: np.sqrt(a**2 - b**2)),
        (("a", "e"), lambda a, e: a * e),
        (("b", "e"), lambda b, e: e * np.sqrt(b**2 / (1 - e**2))),
    )
    E_REQUIREMENTS = (
        (("a", "c"), lambda a, c: c / a),
        (("a", "b"), lambda a, b: np.sqrt(1 - (b/a)**2)),
        (("b", "c"), lambda b, c: c / np.sqrt(b**2 + c**2)),
    )
    FOCUS_REQUIREMENTS = (
        (("a", "b", "angle", "center"), lambda a, b, angle, center: center + rotmat2d(angle) @ np.array([np.sqrt(a**2 - b**2), 0])),
    )
    CENTER_REQUIREMENTS = (
        (("a", "b", "angle", "focus"), lambda a, b, angle, focus: focus - rotmat2d(angle) @ np.array([np.sqrt(a**2 - b**2), 0])),
    )
    V_REQUIREMENTS = (
        (("theta", "a", "b", "mu", "angle"), lambda theta, a, b, mu, angle: velocity_at(theta, a, b, mu, angle)),
    )
    
    def __init__(self, mu, theta, angle, *, a=None, b=None, c=None, e=None, v=None, focus=None, center=None):
        self.a = a
        self.b = b
        self.e = e
        self.c = c
        self.theta = theta
        self.angle = angle
        self.mu = mu
        self.v = v
        self.focus = focus
        self.center = center
        
        for attr in ("a", "b", "e", "v", "focus", "center"):
            if getattr(self, attr) is None:
                found_requirements = False
                for req_list, req_func in getattr(self, f"{attr}_requirements".upper()):
                    suitable = True
                    for requirement in req_list:
                        if getattr(self, requirement) is None:
                            suitable = False
                            break
                    if suitable:
                        setattr(self, attr, req_func(**{requirement: getattr(self, requirement) for requirement in req_list}))
                        found_requirements = True
                        break
                if not found_requirements:
                    raise ValueError(f"Couldn't compute '{attr}' from given data")
    

class OrbitSimulator:
    def __init__(self, initial_conditions: OrbitalParameters):
        self._a = [initial_conditions.a]
        self._b = [initial_conditions.b]
        self._e = [initial_conditions.e]
        self._v = [initial_conditions.v]
        self._theta = [initial_conditions.theta]
        self._angle = [initial_conditions.angle]
        self._center = initial_conditions.center
        self._focus = initial_conditions.focus
        self._mu = initial_conditions.mu
        self._r = [orbit_equation(self._theta[-1], self._a[-1], self._e[-1])]
        self._points = [on_ellipse(self._theta[-1], self._focus, self._a[-1], self._b[-1])]
        self._time = [0]
        
        self.solver = scipy.integrate.ode(self._system).set_integrator("dopri5")
        self.solver.set_initial_value(np.array([*self._points[-1], *self._v[-1]]))
             
    def _system(self, t, y):
        accel = self._accel(t, y)
        point = np.array([y[0], y[1]])
        e_r = point - self._focus
        e_r /= np.linalg.norm(e_r)
        accel_g = -e_r * self._mu / np.sum(np.square(self._focus - point))
        return np.array([y[2], y[3], accel[0]+accel_g[0], accel[1]+accel_g[1]])
    
    def iterate(self, ts, accel):        
        self._accel = accel
        self.solver.integrate(self._time[-1]+ts)
        px, py, vx, vy = self.solver.y
        
        self._time.append(self._time[-1]+ts)
        self._v.append(np.array([vx, vy]))
        self._points.append(np.array([px, py]))
        
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from constants import *
    
    initial_conditions = OrbitalParameters(MU_EARTH, 0, 0, a=6378+400, e=0.1, focus=np.array([0, 0]))
    simulator = OrbitSimulator(initial_conditions)
    ts = 1
    
    def accel(_, y):
        point = np.array([y[0], y[1]])
        e_r = point - initial_conditions.focus
        e_r /= np.linalg.norm(e_r)
        return e_r * 0.0005
    
    # Create a figure
    fig = plt.figure()
    ax = plt.axes(xlim=[-50000, 50000], ylim=[-50000, 50000])
    # ax.axis("equal")
    
    # Initialize a scatter object for the plot
    scatter = ax.scatter([], [], s=1, c="r")
    last = ax.scatter([], [], s=2, c="k")
    foc = ax.scatter(0, 0, s=4, c="b")
    
    def init():
        return scatter, last, foc
    
    # Define the animation function
    def animate(_):
        simulator.iterate(ts, accel)
        scatter.set_offsets(simulator._points[:-1])
        last.set_offsets(simulator._points[-1])
        return scatter, last, foc

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, init_func=init, blit=True, interval=0)
    plt.show()
        