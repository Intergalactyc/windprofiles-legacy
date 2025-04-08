import math
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as st

TRANSFORMS = {
    'linear' : (lambda x : x),
    'log' : (lambda x : np.log(x)),
    'exp' : (lambda x : np.exp(x)),
    'inv' : (lambda x : 1/x),
    'square' : (lambda x : x**2)
}

def ls_linear_fit(xvals, yvals):
    """
    Least squares fit to a relationship y = a + b*x
    Outputs a pair a,b describing fit
    """
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

def power_fit(xvals, yvals, require = 2):
    """
    Least squares fit to relationship y = a*x^b
    Outputs a pair a,b describing fit
    The b is exactly the wind shear coefficient for wind p.l. fit
    """
    xconsider = []
    yconsider = []
    for x,y in zip(xvals, yvals):
        if not (math.isnan(x) or math.isnan(y)):
            if y == 0:
                return 0, np.nan
            xconsider.append(x)
            yconsider.append(y)
    if len(yconsider) < require:
        return np.nan, np.nan
    lnA, B = ls_linear_fit(np.log(xconsider),np.log(yconsider))
    return np.exp(lnA), B

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D # A: amplitude, B: period, C: normalized phase shift, D: offset

def fit_sine(x, y, yerrs, guess_period = 2*np.pi/24, guess_shift = np.pi/2, fix_period = False):

    # tried making a goodness of fit chi^2 test but I don't understand enough
    x = np.array(x)
    y = np.array(y)
    yerrs = np.array(yerrs)

    guess_offset = np.mean(y)
    guess_amplitude = 3 * np.std(y) / np.sqrt(2)

    fitting_function = (lambda t, a, c, d : sine_function(t, a, guess_period, c, d)) if fix_period else sine_function
    guess = [guess_amplitude, guess_shift, guess_offset] if fix_period else [guess_amplitude, guess_period, guess_shift, guess_offset]

    params, pcov = curve_fit(fitting_function, x, y, sigma = yerrs, p0 = guess)
    
    params = [params[0], guess_period, params[1], params[2]] if fix_period else params
    bestfit = lambda t : sine_function(t, *params)

    # Dinv = np.diag(1 / np.sqrt(np.diag(pcov)))
    # pcorr = Dinv @ pcov @ Dinv
    # perrs = np.sqrt(np.diag(pcov))
    # res = ((y - bestfit(x))**2)
    # print(res)
    # chisquared = np.sum(res/(bestfit(x)**2))
    # dof = len(y) - 4 + int(fix_period)
    # pval = stats.distributions.chi2.sf(chisquared, dof)

    # fitinfo = {
    #     'parameters' : params,
    #     'covariances' : pcov,
    #     'correlations' : pcorr,
    #     'errors' : perrs,
    #     'chi_squared' : chisquared,
    #     'p' : pval,
    #     'info' : '`parameters` is best fit parameters a, (b), c, d to a*sin(bx+c)+d'
    # }

    return bestfit, params#fitinfo

def weibull_pdf(x, shape, scale):
    # for x, shape, scale > 0
    return (shape/scale) * (x/scale)**(shape-1) * np.exp(-(x/scale)**shape)

def fit_wind_weibull(data):
    _, shape, _, scale = st.exponweib.fit(data, floc = 0, f0 = 1)
    bestfit = lambda x : weibull_pdf(x, shape, scale)
    return bestfit, [shape, scale]

def rcorrelation(df, col1, col2, transform = ('linear', 'linear')):
    tran_x = TRANSFORMS[transform[0]]
    tran_y = TRANSFORMS[transform[1]]
    dfr = df[~(np.isnan(tran_x(df[col1]))|np.isnan(tran_y(df[col2])))]
    cor = st.pearsonr(tran_x(dfr[col1]), tran_y(dfr[col2]))[0]
    return float(cor)
