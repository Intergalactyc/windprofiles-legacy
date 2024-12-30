# lots of functions for intermediate calculations

import numpy as np
import pandas as pd
import math

LOCAL_GRAVITY = 9.802 # local gravity at Cedar Rapids (latitude ~ 42 degrees, elevation ~ 247 m), in m/s^2

k = 0.4 # von Karman constant. Between 0.35-0.42, some use 0.35 while Stull uses 0.4
z = 2. # sensor at 2 meters
# note - for fluxes, taking theta_v ~ T locally (set reference = values @ 2 m)

# coefficients for dimensionless wind shear computation
ALPHA = 4.7
BETA = 15.

def obukhov_length(u_star, T_bar, Q_0, g = LOCAL_GRAVITY):
    # u_start is friciton velocity, T_bar is mean temperature,
    # Q_0 is mean turbulent vertical heat flux
    L = -u_star**3 * T_bar / (k * g * Q_0)
    return L

def phi(z_over_L):
    # Dimensionless wind shear function
    if z_over_L >= 0: # stable
        # case z/L == 0 is neutral, returns 1 in either formula
        return 1 + ALPHA * z_over_L
    # otherwise, unstable
    return (1 - BETA * z_over_L)**(-1/4)

def wind_gradient(u_star, T_bar, Q_0):
    # uses Businger-Dyer relationship to estimate the vertical gradient of horizontal wind speed, du/dz
    # assume u is aligned with the mean horizontal wind direction
    L = obukhov_length(u_star, T_bar, Q_0)
    grad = u_star / (k * z) * phi(z / L)
    return grad

def flux_richardson(eddy_momt_flux, mean_T, eddy_heat_flux, u_star, g = LOCAL_GRAVITY, report_gradient = False):
    windgrad = wind_gradient(u_star, mean_T, eddy_heat_flux)
    Rif = (g / mean_T) * eddy_heat_flux / (eddy_momt_flux * windgrad)
    if report_gradient:
        return Rif, windgrad
    return Rif

def saturation_vapor_pressure(T):
    # SVP in kPa, using Tetens' approximation
    return 0.6113 * np.exp(17.2694 * (T - 273.15) / (T - 35.86))

def virtual_potential_temperature(RH, P, T):
    # given RH in [0,1], P in kPa, T in K, computes v.p.t.
    e_s = saturation_vapor_pressure(T)
    e = RH * e_s # actual vapor pressure
    mwr = 0.622 # molecular weight ratio of water to air
    w = mwr * e / (P - e) # mixing ratio
    P0 = 100. # reference pressure in kPa
    vT = T * (P0/P)**0.286 # virtual temperature
    vpT = vT * (1+(w/mwr)/(1+w)) # virtual potential temperature, exact form
    # note: there's also an approximation of vpT = vT * (1+0.61*w),
        # which is a first order approximation of the exact formula
        # and is valid within ~1% for mixing ratios between roughly 0.00-0.20
    return vpT

"""
def alternate_virtual_potential_temperature(SH, P, T):
    # given specific humidity in g/m^3, P in kPa, T in K, computes v.p.t.
    r_air = 287.05 # specific gas constant for dry air in J/(kg*K)
    mwr = 0.622 # molecular weight ratio of water to air
    rho_air = P / (r_air * T) # density of air in kg/m^3
    w = SH / rho_air # mixing ratio
    P0 = 100. # reference pressure in kPa
    vT = T * (P0/P)**0.286 # virtual temperature
    vpT = vT * (1+(w/mwr)/(1+w)) # virtual potential temperature, exact form
    print(f'{rho_air=},{w=},{vT=},{vpT=}')
    return vpT
"""

def wind_components(speed, direction, invert=False):
    # given a wind speed and a direction in degrees CW of N,
    # return u, v (eastward, northward) components of wind
    if math.isnan(direction) or speed == 0.:
        return 0., 0.
    direction_rad = np.radians(direction)
    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)
    if invert:
        u *= -1
        v *= -1
    return u, v

def polar_wind(u, v):
    # given u, v (east, north) components of wind,
    # return wind speed, direction 
    speed = np.sqrt(u*u+v*v)
    direction = (np.rad2deg(np.arctan2(u,v)) + 360) % 360
    return speed, direction

def polar_average(magnitudes, directions):
    # polar vector average (true-average)
    radians = np.deg2rad(directions)
    xs = magnitudes * math.cos(radians)
    ys = magnitudes * math.sin(radians)
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    return polar_wind(xavg, yavg)

def angular_distance(alpha, beta):
    # returns unsigned angular distance between alpha and beta
    d0 = (alpha-beta) % 360
    return min(360-d0, d0)

def top_cond_avg(s1,s2,d1,d2,width=30):
    # Given speeds and uncorrected directions for booms 6 and 7 (6 in s1,d1, 7 in s2,d2),
    # finds the best choice of wind speed/direction at 106m:
    # if wind is coming from the east, boom 6 (which is on the west side) is shadowed and so use boom 7,
    # and if from the west, boom 7 (on east side) is shadowed and so use boom 6,
    # and if neither then take the vector average between the two
    # Width is the full width of the east/west bins
    # (e.g. width=30 means any angle within 15 deg of 90/270 for respective boom is flagged)
    hwidth = width / 2
    BOUNDS = [180,360] # UNcorrected angles from which there is shadowing (for boom 6 and 7 respectively)
    d1d = (d1-BOUNDS[0]) % 360
    d2d = (d2-BOUNDS[1]) % 360
    if min(360-d1d, d1d) < hwidth: # if boom 6 is being shadowed (wind from E/90 deg, which is 180 uncorrected), ignore it
        return s2, d2
    elif min(360-d2d, d2d) < hwidth: # if boom 7 is being shadowed (wind from W/270 deg, which is 360 uncorrected), ignore it
        return s1, d1
    else: # if neither is shadowed, vector average both
        x1, y1 = wind_components(s1, d1)
        x2, y2 = wind_components(s2, d2)
        xA = (x1+x2)/2
        yA = (y1+y2)/2
        sA = np.sqrt(xA*xA+yA*yA)
        dA = (np.rad2deg(np.arctan2(xA,yA)) + 360) % 360
        return sA, dA

def combine_top(df,width=30):
    dfC = df.copy(deep=True) # is this necessary? I don't think so because the apply below isn't in place
    res = dfC.apply(lambda row : top_cond_avg(row['ws_106m1'],row['ws_106m2'],row['wd_106m1'],row['wd_106m2'],width), axis=1)
    ws = [r[0] for r in list(res)]
    wd = [r[1] for r in list(res)]
    return ws, wd

def bulk_richardson_number(vpt1, vpt2, z1, z2, ws1, ws2, wd1, wd2):
    # computes the bulk Richardson number Ri given data at two heights z1 and z2
    # vpt is virtual potential temperature, z is height, ws is wind speed, wd is wind direction
    g = LOCAL_GRAVITY
    delta_vpt = vpt2 - vpt1
    delta_z = z2 - z1
    u1, v1 = wind_components(ws1, wd1)
    u2, v2 = wind_components(ws2, wd2)
    delta_u = u2 - u1
    delta_v = v2 - v1
    if delta_u == 0 and delta_v == 0:
        return np.nan
    vpt_avg = (vpt1 + vpt2) / 2
    ri = g * delta_vpt * delta_z / (vpt_avg * (delta_u * delta_u + delta_v * delta_v))
    return ri

def stability_class(Ri):
    if Ri < -0.1:
        return 'unstable'
    if -0.1 <= Ri < 0.1:
        return 'neutral'
    if 0.1 <= Ri < 0.25:
        return 'stable'
    return 'strongly stable'

def stability_class_3(Ri):
    if Ri < -0.1:
        return 'unstable'
    if -0.1 <= Ri < 0.1:
        return 'neutral'
    return 'stable'

def new_class(rat):
    if rat < -0.004:
        return 'unstable'
    if -0.004 <= rat < 0.004:
        return 'neutral'
    return 'stable'

def terrain_class(direction, radius = 15, complex_center = 315, open_center = 135):
    if 2 * radius > angular_distance(complex_center, open_center):
        print('warning: terrain class radius is greater than half the difference between regions.')
    if angular_distance(direction, complex_center) < radius:
        return "complex" # northwest = complex
    elif angular_distance(direction, open_center) < radius:
        return "open" # southeast = open
    else:
        return "other"

def ls_linear_fit(xvals, yvals):
    # Least squares fit to a relationship y = a + b*x
    # Outputs a pair a,b describing fit
    if len(yvals) == 0 or len(xvals) == 0:
        return 0,0
    xvals = list(xvals)
    yvals = list(yvals)
    if len(yvals) != len(xvals):
        raise RuntimeError("Lists must be of equal size")
    for x, y in zip(xvals, yvals):
        if math.isnan(y):
            xvals.remove(x)
            yvals.remove(y)
    n = len(xvals)
    sum_x = sum(xvals)
    sum_x2 = sum(x*x for x in xvals)
    sum_xy = sum(xvals[i]*yvals[i] for i in range(n))
    sum_y = sum(yvals)
    det = n * sum_x2 - sum_x * sum_x
    A = (sum_y * sum_x2 - sum_x * sum_xy)/det
    B = (n * sum_xy - sum_x * sum_y)/det
    return A, B

def power_fit(xvals, yvals, both=False, require = 2):
    # Least squares fit to relationship y = a*x^b
    # Outputs a pair a,b describing fit
    # The b is exactly the wind shear coefficient for wind p.l. fit
    xconsider = []
    yconsider = []
    for x,y in zip(xvals, yvals):
        if not (math.isnan(x) or math.isnan(y)):
            xconsider.append(x)
            yconsider.append(y)
    if len(yconsider) < require:
        if both:
            return np.nan, np.nan
        return np.nan
    lnA, B = ls_linear_fit(np.log(xconsider),np.log(yconsider))
    if both:
        return np.exp(lnA), B
    return B

def seconds(deltatime):
    return deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.microseconds/1e6

def coriolis(deglat, Omega = 7.2921e-5): # returns coriolis parameter in rad/s at a given latitude in degrees
    radlat = np.deg2rad(deglat)
    return 2 * Omega * np.sin(radlat)

def mytest(u0=1.,v0=1.):
    print(f'u = {u0:.3f}, v = {v0:.3f}')
    speed, direction = polar_wind(u0,v0)
    print(f'polar: speed = {speed:.3f}, direction = {direction:.3f} degrees CW of N')

if __name__ == '__main__':
    #mytest()
    pass
