"""Implementation of an SEIR compartment model with R = Recovered + Deceased,
I = I1 + I2 + I3 (increasing severity of infection), with an asymptomatic
infectious compartment (A).
"""

# standard modules
import datetime

# 3rd party modules
import numpy as np
import pandas as pd
from scipy.integrate import odeint


def dataframe_ify(data, start, end, steps):
    """Return human-friendly dataframe of model results.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    start : type
        Description of parameter `start`.
    end : type
        Description of parameter `end`.
    steps : type
        Description of parameter `steps`.

    Returns
    -------
    type
        Description of returned object.

    """

    last_period = start + datetime.timedelta(days=(steps - 1))

    timesteps = pd.date_range(
        # start=start, end=last_period, periods=steps, freq=='D',
        start=start,
        end=last_period,
        freq="D",
    ).to_list()

    # TODO add asymp compartment
    sir_df = pd.DataFrame(
        zip(data[0], data[1], data[2], data[3], data[4], data[5]),
        columns=[
            "exposed",
            "infected_a",
            "infected_b",
            "infected_c",
            "recovered",
            "dead",
        ],
        index=timesteps,
    )

    # resample the values to be daily
    sir_df.resample("1D").sum()

    # drop anything after the end day
    sir_df = sir_df.loc[:end]

    return sir_df


# The SEIR model differential equations.
# TODO update to include asymp compartment
# https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
# but these are the basics
# y = initial conditions
# t = a grid of time points (in days) - not currently used, but will be for time-dependent functions
# N = total pop
# beta = contact rate
# gamma = mean recovery rate
# Don't track S because all variables must add up to 1
# include blank first entry in vector for beta, gamma, p so that indices align in equations and code.
# In the future could include recovery or infection from the exposed class (asymptomatics)
def deriv(y0, t, beta, alpha, gamma, rho, mu, N):
    """Short summary.

    Parameters
    ----------
    y0 : type
        Description of parameter `y0`.
    t : type
        Description of parameter `t`.
    beta : type
        Description of parameter `beta`.
    alpha : type
        Description of parameter `alpha`.
    gamma : type
        Description of parameter `gamma`.
    rho : type
        Description of parameter `rho`.
    mu : type
        Description of parameter `mu`.
    N : type
        Description of parameter `N`.

    Returns
    -------
    type
        Description of returned object.

    """
    dy = [0, 0, 0, 0, 0, 0]
    S = np.max([N - sum(y0), 0])

    dy[0] = np.min([(np.dot(beta[1:3], y0[1:3]) * S), S]) - (alpha * y0[0])  # Exposed
    dy[1] = (alpha * y0[0]) - (gamma[1] + rho[1]) * y0[1]  # Ia - Mildly ill
    dy[2] = (rho[1] * y0[1]) - (gamma[2] + rho[2]) * y0[2]  # Ib - Hospitalized
    dy[3] = (rho[2] * y0[2]) - ((gamma[3] + mu) * y0[3])  # Ic - ICU
    dy[4] = np.min([np.dot(gamma[1:3], y0[1:3]), sum(y0[1:3])])  # Recovered
    dy[5] = mu * y0[3]  # Deaths

    return dy


# Sets up and runs the integration
# start date and end date give the bounds of the simulation
# pop_dict contains the initial populations
# beta = contact rate
# gamma = mean recovery rate
# TODO: add other params from doc
def seir(
    pop_dict, model_parameters, beta, alpha, gamma, rho, mu,
):

    # number of individuals in simulation is equal to total population param
    """Short summary.

    Parameters
    ----------
    pop_dict : type
        Description of parameter `pop_dict`.
    model_parameters : type
        Description of parameter `model_parameters`.
    beta : type
        Description of parameter `beta`.
    alpha : type
        Description of parameter `alpha`.
    gamma : type
        Description of parameter `gamma`.
    rho : type
        Description of parameter `rho`.
    mu : type
        Description of parameter `mu`.

    Returns
    -------
    type
        Description of returned object.

    """
    N = pop_dict["total"]

    # assume that the first time you see an infected population it is mildly so
    # after that, we'll have them broken out
    # TODO add initial condition for asymp compartment when added
    if "infected_b" in pop_dict:
        mild = pop_dict["infected_a"]
        hospitalized = pop_dict["infected_b"]
        icu = pop_dict["infected_c"]
    else:
        hospitalized = pop_dict["infected"] / 4
        mild = hospitalized / model_parameters["hospitalization_rate"]
        icu = hospitalized * model_parameters["hospitalized_cases_requiring_icu_care"]

    exposed = model_parameters["exposed_infected_ratio"] * mild

    susceptible = pop_dict["total"] - (
        pop_dict["infected"] + pop_dict["recovered"] + pop_dict["deaths"]
    )

    # define initial conditions vector
    y0 = [
        int(exposed), # E
        int(mild), # I1
        int(hospitalized), # I2
        int(icu), # I3
        int(pop_dict.get("recovered", 0)), # R
        int(pop_dict.get("deaths", 0)), # D
    ]

    # model step count is equal to number of days to simulate
    steps = 365

    t = np.arange(0, steps, 1)

    ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, N))

    return np.transpose(ret), steps, ret


# Core parameters currently based on the Harvard model.
# for now just implement Harvard model, in the future use this to change
# key params due to interventions
def generate_epi_params(model_parameters):
    """Short summary.

    Parameters
    ----------
    model_parameters : type
        Description of parameter `model_parameters`.

    Returns
    -------
    type
        Description of returned object.

    """
    N = model_parameters["population"]

    fraction_critical = (
        model_parameters["hospitalization_rate"]
        * model_parameters["hospitalized_cases_requiring_icu_care"]
    )

    fraction_severe = model_parameters["hospitalization_rate"] - fraction_critical

    alpha = 1 / model_parameters["presymptomatic_period"]

    # assume hospitalized don't infect
    beta = [
        0,
        model_parameters["beta"] / N,
        model_parameters["beta_hospitalized"] / N,
        model_parameters["beta_icu"] / N,
    ]

    # have to calculate these in order and then put them into arrays
    gamma_0 = 0
    gamma_1 = (1 / model_parameters["duration_mild_infections"]) * (
        1 - model_parameters["hospitalization_rate"]
    )

    rho_0 = 0
    rho_1 = (1 / model_parameters["duration_mild_infections"]) - gamma_1
    rho_2 = (1 / model_parameters["hospital_time_recovery"]) * (
        (fraction_severe + fraction_critical)
    )

    gamma_2 = (1 / model_parameters["hospital_time_recovery"]) - rho_2

    mu = (1 / model_parameters["icu_time_death"]) * (
        model_parameters["case_fatality_rate"] / fraction_critical
    )
    gamma_3 = (1 / model_parameters["icu_time_death"]) - mu

    seir_params = {
        "beta": beta,
        "alpha": alpha,
        "gamma": [gamma_0, gamma_1, gamma_2, gamma_3],
        "rho": [rho_0, rho_1, rho_2],
        "mu": mu,
    }

    return seir_params


# Calculate R0 from differential equations (theoretical)
# TODO update this calculation to include asymptomatic compartment
def generate_r0(seir_params, N):
    """Short summary.

    Parameters
    ----------
    seir_params : type
        Description of parameter `seir_params`.
    N : type
        Description of parameter `N`.

    Returns
    -------
    type
        Description of returned object.

    """
    b = seir_params["beta"]
    p = seir_params["rho"]
    g = seir_params["gamma"]
    u = seir_params["mu"]

    r0 = N * (
        (b[1] / (p[1] + g[1]))
        + (p[1] / (p[1] + g[1]))
        * (b[2] / (p[2] + g[2]) + (p[2] / (p[2] + g[2])) * (b[3] / (u + g[3])))
    )

    return r0


# Credit: http://code.activestate.com/recipes/579103-python-addset-attributes-to-list/
class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """
    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
