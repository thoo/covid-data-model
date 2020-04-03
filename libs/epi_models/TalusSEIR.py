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


def brute_force_r0(seir_params, new_r0, r0, N):
    """Given a target R0 `new_r0`, tune the value of beta_1 until the value of
    R0 calculated from the model parameters is equal to the target R0 `new_r0`,
    updates the SEIR parameters dictionary to include the updated beta_1, and
    returns the updated SEIR parameters dictionary.

    This procedure is used to simulate the effect of interventions that are
    assumed to reduce R0 from the time they're applied until a specified future
    time. The timeframe of interventions and the R0 value they are assumed to
    yeild are defined in driver code (e.g., run.py).

    NOTE: This function will be obsolete when the procedure for introducing
    interventions into model runs is updated -- do not maintain it.

    Parameters
    ----------
    seir_params : dict
        Dictonary of model parameters, including beta, which is tuned in this
        function until the calculated R0 equals `new_r0`.
    new_r0 : float
        The value of R0 that a particular intervention is assumed to yeild.
    r0 : float
        The current value of R0.
    N : int
        The total population in the simulation.

    Returns
    -------
    dict
        Updated SEIR parameter dictionary including the updated value of beta_1
        that yeilded `new_r0`.

    """
    # init the calculated R0 equal to the current R0
    calc_r0 = r0

    # initially, "nudge" the current value of beta_1 by a small epsilon,
    # hopefully closer toward a value that will yeild the target value of R0
    # `new_r0`.
    change = np.sign(new_r0 - calc_r0) * 0.00005

    # init the new SEIR parameter dictionary as a copy of the original; these
    # will be returned by this func at the end.
    new_seir_params = seir_params.copy()

    # continue iterations until the calculated R0 is very close to the target
    # r0 `new_r0`.
    while round(new_r0, 4) != round(calc_r0, 4):

        # nudge beta_1 in a direction that will bring the calculated R0
        # closer to the `new_r0`.
        new_seir_params["beta"] = [
            0.0,
            new_seir_params["beta"][1] + change,
            new_seir_params["beta"][2],
            new_seir_params["beta"][3],
        ]

        # get the new calculated R0 value using the theoretical solution for it
        # for this model
        calc_r0 = generate_r0(new_seir_params, N)

        # if the sign of the difference between the target and calculated R0
        # has changed, halve the size of the "nudge" and change its direction,
        # since the algorithm overshot past the target R0.
        diff_r0 = new_r0 - calc_r0
        if np.sign(diff_r0) != np.sign(change):
            change = -change / 2

        # otherwise, leave the value of `change` alone and keep iterating

    # return the updated dictionary of SEIR parameters including the updated
    # value of beta_1 that yeilds `new_r0`.
    return new_seir_params


def dataframe_ify(data, start, end, steps):
    """Convert model run data into a DataFrame for viewing and exploration.

    Parameters
    ----------
    data : list of lists
        Output of odeint (see function `seir` below), which is a list of lists:
        one list per modeled timestep containing the value of each integrated
        equation at that timestemp.
    start : datetime.date
        Date of the start of the simulation.
    end : datetime.date
        NOT CURRENTLY USED. The end date is automatically calculated by adding
        a number of days equal to the number of simulation timesteps minus 1
        to the start argument.
    steps : type
        Description of parameter `steps`.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the results of the simulation in human-friendly
        format.

    """
    # calculate end date of dataframe based on start date so the number of days
    # within the timeframe is identical to the number of timesteps in the
    # simulation.
    end = start + datetime.timedelta(days=(steps - 1))

    # get list of timesteps (dates between the start and end dates)
    timesteps = pd.date_range(
        start=start,
        end=end,
        freq="D",
    ).to_list()

    # zip up data from the ODE integrator solution output into a human-readable
    # dataframe with column names
    sir_df = pd.DataFrame(
        zip(data[0], data[1], data[2], data[3], data[4], data[5], data[6]),
        columns=[
            "exposed",
            "infected_a",
            "infected_b",
            "infected_c",
            "recovered",
            "dead",
            "asymp",
        ],
        index=timesteps,
    )

    # resample the values to be daily, in case they aren't already
    sir_df.resample("1D").sum()

    # ensure no data beyond the end day are returned
    sir_df = sir_df.loc[:end]

    # return dataframe containing the results of the simulation in
    # human-friendly format
    return sir_df


def deriv(y0, t, beta, alpha, gamma, rho, mu, f, N):
    """Calculate and return the current values of dE/dt, etc. for each model
    compartment as numerical integration is performed. This function is the
    first argument of the odeint numerical integrator function.

    Parameters
    ----------
    y0 : list
        The values of the equations being integrated at X(t) where X is the
        equation being integrated at timestep t.
    t : float
        The current value of time.
    beta : list of floats
        Transmission rates (per day) for various classes of infectious people.
    alpha : float
        Rate (per day) at which exposed (E) individuals become infectious (I_1).
    gamma : list of floats
        Rates (per day) at which various classes of infectious people recover.
    rho : list of floats
        Rates (per day) at which various classes of infectious people develop
        the next severity level of symptoms
    mu : float
        Rate (per day) at which infectious people with critical symptoms (I_3)
        pass away.
    N : int
        The total number of individuals in the simulation.

    Returns
    -------
    list of floats
        Description of returned object.

    """
    # susceptible (S) population is equation to the total population N minus
    # the population of all other classes, and must be greater than zero.
    S = np.max([N - sum(y0), 0])

    # define human-readable equation terms
    # state equation variables
    E = y0[0]
    I1 = y0[1]
    I2 = y0[2]
    I3 = y0[3]
    R = y0[4]
    D = y0[5]
    A = y0[6]

    # other terms
    I_all = [I1, I2, I3]
    I_transmission = np.dot(beta[1:4], I_all)
    I_recovery = np.dot(gamma[1:4], I_all)
    A_transmission = A * beta.A
    A_recovery = A * gamma.A
    all_infected = sum(I_all) + A

    # state equations
    dE = np.min([(A_transmission + I_transmission) * S, S]) - (alpha * E)  # Exposed
    dA = ((1 - f) * alpha * E) - (gamma.A * A)  # asymp
    dI1 = (f * alpha * E) - (gamma[1] + rho[1]) * I1  # Ia - Mildly ill
    dI2 = (rho[1] * I1) - (gamma[2] + rho[2]) * I2  # Ib - Hospitalized
    dI3 = (rho[2] * I2) - ((gamma[3] + mu) * I3)  # Ic - ICU
    dR = np.min([A_recovery + I_recovery, all_infected])  # Recovered
    dD = mu * I3  # Deaths

    # return a vector of the current deltas for each equation being integrated
    dy = [dE, dI1, dI2, dI3, dR, dD, dA]
    return dy


def seir(pop_dict, model_parameters, beta, alpha, gamma, rho, mu, f):
    """Given various input parameters including initial populations and model
    rate constants, performs a model run of the SEIR model, and returns the
    values of the integrated differential equations for each simulated timestep.

    Parameters
    ----------
    pop_dict : dict
        Dictionary containing initial population values. If the populations of
        the three types of infected classes are known, they should be defined
        with keys "infected_a", "infected_b", and "infected_c". These other
        populations should also be defined:
            "total". The total population.
            "infected". The initial infected (I_1 + I_2 + I_3) population.
            "recovered". The initial recovered (R) population.
            "deaths". The initial deceased (D) population.
    model_parameters : dict
        Dictionary of model parameters, including all physical/empirical values
        needed to define all rate constants and the value of N (total pop.) used
        in the SEIR model.
    beta : list of floats
        Transmission rates (per day) for various classes of infectious people.
    alpha : float
        Rate (per day) at which exposed (E) individuals become infectious (I_1).
    gamma : list of floats
        Rates (per day) at which various classes of infected people recover.
    rho : list of floats
        Rates (per day) at which various classes of infected people develop
        the next severity level of symptoms
    mu : float
        Rate (per day) at which infectious people with critical symptoms (I_3)
        pass away.

    Returns
    -------
    list
        Element 1: values of integrated differential equations at each timestep
                   (transposed)
        Element 2: timestemps over which numerical integration was performed
        Element 3: values of integrated differential equations at each timestep
                   (not transposed)

    """
    # total population to be simulated
    N = pop_dict["total"]

    # if the initial populations of the three classes of infected people have
    # been defined, use them. These values may have been defined if this
    # simulation is resuming from where a prior simulation left off, e.g., to
    # simulate the effect of an intervention at a certain timestep.
    infected_pop_init_conditions_defined = "infected_b" in pop_dict
    if infected_pop_init_conditions_defined:
        mild = pop_dict["infected_a"]
        hospitalized = pop_dict["infected_b"]
        icu = pop_dict["infected_c"]
        asymp = pop_dict["asymp"]

    # otherwise, if only the total number of infected people is known and not
    # the specific number in each of the three severity classes, estimate the
    # number of people in each severity class using model parameters and that
    # total number of infected people.
    else:
        hospitalized = pop_dict["infected"] / 4
        mild = hospitalized / model_parameters["hospitalization_rate"]
        icu = hospitalized * model_parameters["hospitalized_cases_requiring_icu_care"]
        asymp = mild * model_parameters["asymp_to_mild_ratio"]

    # let the number of susceptible people be any one not currently infected,
    # recovered, or deceased
    susceptible = pop_dict["total"] - (
        pop_dict["infected"] + pop_dict["recovered"] + pop_dict["deaths"]
    )

    # obtain number of exposed people at simulation start from the number of
    # people in class I_1 (sick with mild symptoms)
    exposed = model_parameters["exposed_infected_ratio"] * mild \
        if model_parameters["exposed_from_infected"] \
        else pop_dict.get("exposed", 0)

    y0 = [
        int(exposed),
        int(mild),
        int(hospitalized),
        int(icu),
        int(pop_dict.get("recovered", 0)),
        int(pop_dict.get("deaths", 0)),
        int(asymp),
    ]

    # define number of timesteps to simulate and create a 0-indexed list of
    # timestep indices 0..N where N = steps below
    steps = 365
    t = np.arange(0, steps, 1)

    # get values of integrated differential equations at each timestep
    ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, f, N))
    return np.transpose(ret), steps, ret


def generate_epi_params(model_parameters):
    """Given a dictionary of model parameters including some physical/empirical
    values and possibly some rate constants, return a dictionary of the full set
    of rate constants required to numerically integrate the SEIR model.

    For additional information and references relating to these calculations,
    see https://alhill.shinyapps.io/COVID19seir/ >> "Model" section.

    Parameters
    ----------
    model_parameters : dict
        Dictionary of model parameters, including all physical/empirical values
        needed to define all rate constants and the value of N (total pop.) used
        in the SEIR model.

    Returns
    -------
    dict
        Dictionary of SEIR parameters, all rate constants used in the model.

    """
    # N: total population.
    N = model_parameters["population"]

    ############################################################################
    # ALPHA: rate (per day) at which exposed (E)individuals become
    # infectious (I_1). Calculated as the reciprocal of the incubation
    # period (days).
    ############################################################################
    alpha = 1 / model_parameters["presymptomatic_period"]

    ############################################################################
    # BETA: transmission rates (per day) for various classes of
    # infectious people. Currently determined directly from model parameters.
    ############################################################################
    beta = L(
        # define placeholder for gamma list's zeroth element so human-readable
        # indices can be used in calculations
        0,
        model_parameters["beta"] / N,
        model_parameters["beta_hospitalized"] / N,
        model_parameters["beta_icu"] / N,
        A=model_parameters["beta_asymp"] / N,
    )

    ############################################################################
    # GAMMA: rates (per day) at which various classes of infected and infectious
    # people recover.
    ############################################################################
    # define placeholder for list's zeroth element so human-readable
    # indices can be used in calculations
    gamma_0 = 0

    # gamma_A: for class A
    gamma_A = (1 / model_parameters["duration_asymp_infections"])

    # gamma_1: for class I_1
    gamma_1 = (1 / model_parameters["duration_mild_infections"]) * (
        1 - model_parameters["hospitalization_rate"]
    )

    # NOTE: gamma_2 depends on rho_2 and is calculated in the rho section

    # fraction of infected cases that develop critical symptoms (ICU)
    fraction_critical = (
        model_parameters["hospitalization_rate"]
        * model_parameters["hospitalized_cases_requiring_icu_care"]
    )

    # fraction of infected cases that develop severe symptoms (hospitalized)
    fraction_severe = model_parameters["hospitalization_rate"] \
        - fraction_critical

    ############################################################################
    # RHO: rates (per day) at which various classes of infected people develop
    # the next severity level of symptoms.
    ############################################################################
    # define placeholder for list's zeroth element so human-readable
    # indices can be used in calculations
    rho_0 = 0

    # rho_1: for class I_1
    rho_1 = (1 / model_parameters["duration_mild_infections"]) - gamma_1

    # rho_2: for class I_2
    rho_2 = (1 / model_parameters["hospital_time_recovery"]) * (
        (fraction_critical / (fraction_severe + fraction_critical))
    )

    # gamma_2: for class I_2, and which is calculated from rho_2
    gamma_2 = (1 / model_parameters["hospital_time_recovery"]) - rho_2

    ############################################################################
    # MU: rate (per day) at which infectious people with critical symptoms (I_3)
    # pass away.
    ############################################################################
    mu = (1 / model_parameters["icu_time_death"]) * (
        model_parameters["case_fatality_rate"] / fraction_critical
    )

    # gamma_3: for class I_3, and which is calculated from mu
    gamma_3 = (1 / model_parameters["icu_time_death"]) - mu

    # collate SEIR rate constants into a dictionary of parameters and return it
    seir_params = {
        "beta": beta,
        "alpha": alpha,
        "gamma": L(gamma_0, gamma_1, gamma_2, gamma_3, A=gamma_A),
        "rho": [rho_0, rho_1, rho_2],
        "mu": mu,
        "f": convert_ratio_to_frac(
            model_parameters["asymp_to_mild_ratio"],
            numerator='second_term'
        )
    }
    return seir_params


def generate_r0(seir_params, N):
    """Given SEIR params including those for an asymptomatic, infectious (A)
    class, and the number of people in the population, returns the value of R0
    calculated directly from the differential equations (NOT numerically).

    Parameters
    ----------
    seir_params : dict
        Dictionary of model parameter vectors, including beta, gamma, etc.
    N : int
        Number of people in the population.

    Returns
    -------
    float
        R0 for the model.

    """
    # define parameters and parameter vectors
    f = seir_params["f"]
    b = seir_params["beta"]
    p = seir_params["rho"]
    g = seir_params["gamma"]
    m = seir_params["mu"]

    # define individual variables used in the R0 formula for legibility
    # betas
    BA = b.A
    B1 = b[1]
    B2 = b[2]
    B3 = b[3]

    # gammas
    gA = g.A
    g1 = g[1]
    g2 = g[2]
    g3 = g[3]

    # rhos
    p1 = p[1]
    p2 = p[2]

    # calculate theoretical R0 from model equations solved at steady state
    # conditions and return it
    r0 = N * (
        (1 - f) * BA / gA + \
            f * (
                (B1 / (p1 + g1)) + \
                    (p1 / (p1 + g1)) * \
                        (B2 / (p2 + g2) + (p2 / (p2 + g2)) * (B3 / (m + g3)))
            )
    )
    return r0


def convert_ratio_to_frac(x, numerator='second_term'):
    """Given a ratio x where x = a / b, returns the corresponding fraction
    assuming that b is the numerator by default.

    Parameters
    ----------
    x : float
        ratio where x = a /b.
    numerator : string
        if 'second_term': returns fraction assuming b is numerator.
        if 'first_term': returns fraction assuming a is numerator.

    Returns
    -------
    float
        fraction assuming b is the numerator by default, i.e., b / (a + b).
        if 'first_term' was specified as `numerator` then returns fraction
            assuming a is the numerator, i.e., a / (a + b)

    """
    if numerator == 'second_term':
        return 1 / (x + 1)
    else:
        return x / (x + 1)


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
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


def harvard_model_params(N):
    """For testing only: given the total number of people in the simulation,
    return the default values of the Alison Hill model's parameters, available
    online at https://alhill.shinyapps.io/COVID19seir/.

    Parameters
    ----------
    N : int
        Total number of people in the simulation.

    Returns
    -------
    dict
        Dictionary of model parameter vectors, including beta, gamma, etc.

    """
    return {
        "beta": [0.0, 0.5 / N, 0.1 / N, 0.1 / N],
        "alpha": 0.2,
        "gamma": [0.0, 0.133, 0.125, 0.075],
        "rho": [0.0, 0.033, 0.042],
        "mu": 0.05,
        "f": 1,
    }


def r0_24_params(N):
    """For testing only: given the total number of people in the simulation,
    return the values of the Alison Hill model's parameters that yeild an R0
    of 2.4. The model is available online at
    https://alhill.shinyapps.io/COVID19seir/.

    Parameters
    ----------
    N : int
        Total number of people in the simulation.

    Returns
    -------
    dict
        Dictionary of model parameter vectors, including beta, gamma, etc. that
        will yeild an R0 of 2.4 when used in the Alison Hill model.

    """
    return {
        "beta": [0.0, 0.3719985820912413 / N, 0.1 / N, 0.1 / N],
        "alpha": 0.2,
        "gamma": [0.0, 0.133, 0.125, 0.075],
        "rho": [0.0, 0.033, 0.042],
        "mu": 0.05,
        "f": 1,
    }
