import numpy as np

from constants import *


def radiation_pressure(distance, reflectivness=1):
    """Compute the radiation pressure of the sun on a body.

    Args:
        distance (float): The distance from the body to the sun.
        reflectivness (float, optional): The reflectivness of the body. Defaults to 1.

    Returns:
        float: The applied radiation pressure.
    """
    return AU**2 * ((reflectivness + 1) * G_SC / c) / distance**2

def escape_up(t, y, orbital_params, reflectivness, area, mass):
    point = np.array([y[0], y[1]])
    vel = np.array([y[2], y[3]])
    e_r = point - orbital_params._focus
    r = np.linalg.norm(e_r)
    e_vec = np.cross(vel, np.cross(np.array([*e_r, 0]), vel))[:2] / orbital_params._mu - e_r / r
    if np.cross(e_vec, e_r) > 0:
        e_r /= r
        solar_radiation_accel = 1e-3 * radiation_pressure(r * 1e3, reflectivness) * e_r * area / mass
        return solar_radiation_accel
    return e_r * 0

def escape_earth(t, y, orbital_params, reflectivness, area, mass):
    point = np.array([y[0], y[1]])
    center = point - np.array([-AU*1e-3, 0])
    vel = np.array([y[2], y[3]])
    e_r = point - orbital_params._focus
    r = np.linalg.norm(e_r)
    e_vec = np.cross(vel, np.cross(np.array([*e_r, 0]), vel))[:2] / orbital_params._mu - e_r / r
    if np.cross(e_vec, e_r) > 0:
        sun_e_r = point - center
        sun_r = np.linalg.norm(sun_e_r)
        sun_e_r /= sun_r
        e_r /= r
        solar_radiation_accel = 1e-3 * radiation_pressure(sun_r * 1e3, reflectivness) * sun_e_r * area / mass
        return solar_radiation_accel
    return e_r * 0

def always(t, y, orbital_params, reflectivness, area, mass):
    point = np.array([y[0], y[1]])
    e_r = point - orbital_params._focus
    r = np.linalg.norm(e_r)
    e_r /= r
    solar_radiation_accel = 1e-3 * radiation_pressure(r * 1e3, reflectivness) * e_r * area / mass
    return solar_radiation_accel
    
def tangential(t, y, orbital_params, mag):
    point = np.array([y[0], y[1]])
    e_r = point - orbital_params._focus
    e_t = np.array([-e_r[1], e_r[0]])
    e_t /= np.linalg.norm(e_t)
    return e_t * mag