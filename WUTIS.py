import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

def drawdown(data: pd.DataFrame):
    """
    Takes a time series of asset prices, returns a DataFrame with columns for
    the asset prices, the prior peaks, and the percentage drawdown
    """
    asset_price = data
    prior_peak = asset_price.cummax()
    drawdown = (asset_price - prior_peak) / prior_peak
    max_drawdown = drawdown.cummin()
    return pd.DataFrame({"Asset Price": asset_price, 
                         "Prior Peak": prior_peak, 
                         "Drawdown": drawdown,
                        "Max Drawdown": max_drawdown})

def var_historic(data: pd.DataFrame, level = 99):
    """
    Takes a time series of asset returns, returns the historic
    Value at Risk at a specified confidence level
    """
    return -np.percentile(data, (100 - level))

def cvar_historic(data: pd.DataFrame, level = 99):
    """
    Takes a time series of asset returns, returns the historic
    Conditional Value at Risk at a specified confidence level
    """
    is_beyond = data <= -var_historic(data, level = level)
    return -data[is_beyond].mean()

def var_cornfish(data: pd.DataFrame, level = 99):
    """
    Takes a time series of asset returns, returns the Cornish-Fisher
    modified Value at Risk at a specified confidence level
    """
    # compute normal distribution z-score
    z = st.norm.ppf((100 - level)/100)
    # modify the z-score based on skewness and kurtosis
    s = st.skew(data)
    k = st.kurtosis(data, fisher = False)
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36)
    # estimate the VaR based on modified z-score * std distance from the mean
    return -(data.mean() + z * data.std(ddof=0))

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns given the periods per year
    """
    compounded_growth = (1+r).prod() 
    n_periods = r.shape[0] 
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns given the periods per year
    """
    return r.std()*(periods_per_year**0.5)

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights and returns are an Nx1 matrix
    """
    # @ operator stands for matrix multiplication. 
    # .T returns a transpose of an array
    return weights.T @ returns 

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are an N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0] #number of assets
    init_guess = np.repeat(1/n, n) #equally weighted portfolio as an initial guess for the optimizer
    bounds = ((0.0, 1.0),) * n #a set of 0% to 100% weight bounds for each asset

    #constraint: difference between sum of weigths and 1 is equal to 0 i.e. sum of weights has to be 100%
    #"type":"eq" means equality
    #"fun" stands for a function defining a constraing
    # lambda is a special type of nameless functions in python that can be contained in one line of code
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    #constraint: return is equal to the target return
    #"args" stands for extra arguments
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)}
    #runs the optimizer to minimize the volatility for a given return
    #opt.minimize(the thing to be minimized, initial guess, extra arguments, minimization method (slsqp stands for sequantial least squares), optimization details, ...)
    # (..., constraints (weights add up to 100%, target return is target return), bounds (individual weights are between 0 and 100%))
    weights = opt.minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x #get the argument x

def optimal_weights(n_points, er, cov):
    """
    Calculates optimal porfolio weights for n points between maximum and minimum possible return
    """
    #Get a linear space from min and max return possible of length n_points
    target_rs = np.linspace(er.min(), er.max(), n_points)
    #Get optimal weights for each of n_points returns
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov):
    """
    Plots the multi-asset efficient frontier
    """
    #get weights for each element of the interpolated lin space
    weights = optimal_weights(n_points, er, cov)
    #get return and volatility for each weights pair
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style='.-')

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0] 
    init_guess = np.repeat(1/n, n) 
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    #maximize function does not exist, thus what we will be doing is minimizing the negative sharpe ratio
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    weights = opt.minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio given a covariance matrix
    """
    # I love this function, what we are doing here is we are feeding the MSR optimizer the idea that all the returns are the same and the way it reacts
    # is it starts to increase the SR by lowering volatility and does so until it reaches the point of lowest possible volatility
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def cppi(risky, safe, m = 2, start = 100, floor = 0.75, dynamic = False):
    """
    Runs a backtest of the CPPI strategy, returns 
    a dataframe with Wealth, Floor, % of Wealth in Risky Asset and Cushion as % of Wealth.
    risky -> pandas risky asset returns series
    safe -> pandas safe asset returns series
    m -> multiplier
    start -> starting capital
    floor -> cushion as % of starting capital
    dynamic -> dynamic/non-dynamic floor
    """
    # set up the CPPI parameters
    dates = risky.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value

    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(pd.DataFrame(risky))
    risky_w_history = pd.DataFrame().reindex_like(pd.DataFrame(risky))
    cushion_history = pd.DataFrame().reindex_like(pd.DataFrame(risky))
    floor_history = pd.DataFrame().reindex_like(pd.DataFrame(risky))
    
    # a for loop that does the backtest
    for step in range(n_steps):
    # if dynamic is true, peak is either previous peak or account value, floor value is peak * floor
        if dynamic is True:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (floor)
        # calculations that correspond to the formulas on the picture
        cushion = (account_value - floor_value)/account_value
        risky_w = m * cushion
        # we have to make sure the risky asset allocation is not bigger than 1 (can happen if you have a very big m)
        # and not smaller than 0 (can happen when your total assets value goes below the floor value)
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky.iloc[step]) + safe_alloc*(1+safe.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floor_history.iloc[step] = floor_value
    # combining the results
    backtest_result = pd.concat([account_history, floor_history, risky_w_history, cushion_history], axis=1,
                                keys = ["Wealth","Floor","% of Wealth in Risky Asset","Cushion as % of Wealth"])
    # transforming keys to column names
    backtest_result.columns = backtest_result.columns.get_level_values(0)
    return backtest_result

def mc(n_years = 10, n_scenarios = 100, mu = 0.1, sigma = 0.1, steps_per_year = 52, start = 100.0):
    """
    Generates stock prices using Monte Carlo methodology
    n_years -> number of years
    n_scenarios -> number of scenarios
    mu -> annualized return
    sigma -> annualized volatility
    steps_per_ year -> number of steps per year
    start -> starting wealth
    """
    # calculate the annualization coefficient and the number of steps
    ann = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # calculate the returns for every time period using mu and sigma (loc = mean, scale = standard deviation, size = shape of the output)
    rets_plus_1 = np.random.normal(loc = (1+mu)**ann, scale = (sigma*np.sqrt(ann)), size = (n_steps, n_scenarios))
    # transform returns into prices (make the first row = 1), all the subsequent rows are a cumproduct
    rets_plus_1[0] = 1
    ret_val = start*pd.DataFrame(rets_plus_1).cumprod()
    return ret_val