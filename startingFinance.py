import numpy
from scipy.integrate import quad

def dN(x):
    """Probability density function of standard normal random variables x

    Arguments:
        x {[float]} -- [value]
    """
    value = numpy.exp(-0.5*x**2)/numpy.sqrt(2*numpy.pi)
    return value 

def N(d):
    """ Cumulative density function

    Arguments:
        d {[float]} -- [description]
    """
    value = quad(lambda x : dN(x), -20, d, limit=50)[0]
    return value


def d1f(St, K, t, T, r, sigma):
    """Black scholes merton d1 function

    Arguments:
        St {[float]} -- [stock level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in year fractions)]
        T {[float]} -- [maturity date (in year fractions)]
        r {[float]} -- [constant risk-free short rate]
        sigma {[float]} -- [volatility factor in diffusion term]
    """
    d1 = (numpy.log(St/K) + (r + 0.5*sigma**2)*(T-t))/(sigma*numpy.sqrt(T-t))
    return d1

def bsmCallValue(St, K, t,  T, r, sigma):
    """Calculates Black-Scholes European call option value

    Arguments:
        St {[float]} -- [stock level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date]
        T {[float]} -- [maturity date]
        r {[float]} -- [constant risk-free short rate]
        sigma {[float]} -- [volatility factor in diffusion term]
    """
    d1 = d1f(St,K,t,T,r,sigma)
    d2 = d1 - sigma*numpy.sqrt(T-t)
    callValue = St*N(d1)- numpy.exp(-r*(T-t))*K*N(d2)
    return callValue

def bsmPutValue(St, K, t,  T, r, sigma):
    """Calculates Black-Scholes European put option value

    Arguments:
        St {[float]} -- [stock level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date]
        T {[float]} -- [maturity date]
        r {[float]} -- [constant risk-free short rate]
        sigma {[float]} -- [volatility factor in diffusion term]
    """
    putValue = bsmCallValue(St,K,t,T,r,sigma) - St+numpy.exp(-r*(T-t))*K
    return putValue

def bsmDelta(St, K, t, T, r, sigma):
    """ Black-Scholes-Merton Delta of european call option

    Arguments:
        St {[float]} -- [Stock at level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in years fractions)]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
    
    Returns 
    =========
    European call option DELTA

    """
    d1 = d1f(St, K, t, T, r, sigma)
    delta = N(d1)
    return delta

def bsmGamma(St, K, t, T, r, sigma):
    """Black scholes-Meron Gamma of european call option

    Arguments:
        St {[float]} -- [Stock at level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in years fractions)]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
    
    Returns 
    ========
    gamma {[float]} -- [European call option gamma]
    """
    d1 = d1f(St, K, t, T, r, sigma)
    gamma = dN(d1)/(St * sigma*numpy.sqrt(T-t))
    return gamma

def bsmTheta(St, K, t, T, r, sigma):
    """Black scholes theta of european call option

    Arguments:
        St {[float]} -- [Stock at level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in years fractions)]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
    
    Returns 
    =========
    theta {[float]} -- [European call theta]
    """
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma*numpy.sqrt(T-t)
    theta = -(St*dN(d1)*sigma/(2*numpy.sqrt(T-t)) + r*K*numpy.exp(-r*(T-t))*N(d2))
    return theta


def bsmRho(St, K, t, T, r, sigma):
    """Black Scholes-Merton Rho of europpean call option

    Arguments:
        St {[float]} -- [Stock at level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in years fractions)]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
    """
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma*numpy.sqrt(T-t)
    rho = K*(T-t)*numpy.exp(-r*(T-t))*N(d2)
    return rho

def bsmVega(St, K, t, T, r , sigma):
    """ Black Scholes Merton Vegaof european call option

    Arguments:
        St {[float]} -- [Stock at level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date (in years fractions)]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
    """
    d1 = d1f(St, K, t, T, r, sigma)
    vega = St*dN(d1)*numpy.sqrt(T-t)
    return vega

def crrOptionValue(S0, K, T, r, sigma, otype , M = 4):
    """Cox-Ross rubinstein European option valuation

    Arguments:
        S0 {[float]} -- [stock level at time 0]
        K {[float]} -- [strike price]
        T {[float]} -- [maturity date (in years fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
        otype {[string]} -- [either call or put]

    Keyword Arguments:
        M {int} -- [number of time intervals] (default: {4})
    """
    dt = T/M
    df = numpy.exp(-r*dt)  # discount per interval

    #Binomial parameters
    u = numpy.exp(sigma*numpy.sqrt(dt))
    d = 1/u #down movement
    q = (numpy.exp(r*dt) - d) /(u- d)  #martingale branch probability

    #Array Initialization for index levels
    mu = numpy.arange(M+1)
    mu = numpy.resize(mu, (M+1, M+1))
    md = numpy.transpose(mu)
    mu = u**(mu-md)
    md = d**md
    S = S0*mu*md

    #Inner Values
    if otype == 'call':
        V = numpy.maximum(S - K, 0)
    elif otype == 'put':
        V = numpy.maximum(K-S, 0)
    else:
        raise("Please the option does not exists")
    
    z = 0
    for t in range(M-1, -1, -1): 
        V[0:M-z, t] = (q *V[0:M-z, t+1] + (1-q)*V[1:M-z+1, t+1])*df
        z +=1
    return V[0,0], V 


def monteCarloCall(S0, K, T, r, sigma, path, iters= 1000):
    """Compute the call using monte carlo method

    Arguments:
        S0 {[float]} -- [stock level at time 0]
        K {[float]} -- [strike price]
        T {[float]} -- [maturity date (in year fractions)]
        r {[float]} -- [constant risk-free short date]
        sigma {[float]} -- [volatility factor in diffusion term]
        path {[int]} -- [number of path suggested into simulation]

    Keyword Arguments:
        iter {int} -- [number of observation in time] (default: {1000})
    """
    S = numpy.zeros((path+1, iters))
    S[0] = S0
    for n in range(1, path+1):
        Z = numpy.random.standard_normal(iters)
        S[n] = S[n-1]*numpy.exp((r - 0.5*sigma**2)*path/iters + sigma*numpy.sqrt(path/iters)*Z)
    
    C = numpy.exp(-r*T)*numpy.sum(numpy.maximum(S[-1]-K,0))/iters
    return C, S


def assetParams(**kwargs):
    """asset parameters

    The dictionary contains :
    St  stock level at time t
    K strike price 
    points number of points in discretization
    t  valuation date at time t
    T  maturity date
    r constant risk-free short date
    sigma volatility factor in diffusion term

    Kplot : parametrization of K curve with start and end date
    Tplot : parametrization of T curve with start and end date
    Rplot : parametrization of R curve with start and end date
    Splot : parametrization of S curve with start and end date
    Returns:
    [dict] -- [asset param]
    """
    params= {
            'start' :80 if 'start' not in kwargs else kwargs['start'],
            'end' : 120 if 'endK' not in kwargs else kwargs['end'],
            'points' : 100 if 'points' not in kwargs else kwargs["points"],
            'St' : 100 if 'St' not in kwargs else kwargs["St"],
            'K' : 100 if 'K' not in kwargs else kwargs["K"],
            't' : 0.0 if 'valuationDate' not in kwargs else kwargs["valuationDate"],
            'T' : 1.0 if 'maturityDate' not in kwargs else kwargs["maturityDate"],
            'sigma' :0.2 if 'sigma' not in kwargs else kwargs["sigma"]
            }
    
    if 'Kplot' in kwargs:
        params.update(
            {
                'Kplot':
                {
                    'start': 80 if 'start' not in kwargs["Kplot"] else kwargs["Kplot"]["start"],
                    'end':   120 if 'end' not in kwargs["Kplot"] else kwargs["Kplot"]["end"]
                }
            })

    if 'Rplot' in kwargs:
        params.update(
            {
                'Rplot':
                {
                    'start': 0 if 'start' not in kwargs["Rplot"] else kwargs["Rplot"]["start"],
                    'end':  0.1  if 'end' not in kwargs["Rplot"] else kwargs["Rplot"]["end"]
                }
            })

    if 'Tplot' in kwargs:
        params.update(
            {
                'Tplot':
                {
                    'start': 0.0001 if 'start' not in kwargs["Tplot"] else kwargs["Tplot"]["start"],
                    'end':   1 if 'end' not in kwargs["Tplot"] else kwargs["Tplot"]["end"]
                }
            })

    if 'Splot' in kwargs:
        params.update(
            {
                'Splot':
                {
                   'start': 0.01 if 'start' not in kwargs["Splot"] else kwargs["Splot"]["start"],
                    'end':    0.5 if 'end' not in kwargs["Splot"] else kwargs["Splot"]["end"]
                }
            })
    return params

def bsmImpliedVol(St, K, t, T, r, C0, sigmaEst, iters = 100):
    """Calculation of implied vol

    Arguments:
        St {[float]} -- [stock level at time t]
        K {[float]} -- [strike price]
        t {[float]} -- [valuation date]
        T {[float]} -- [maturity date]
        r {[float]} -- [constant risk-free short date]
        C0 {[float]} -- [call price on market to market]
        sigmaEst {[float]} -- [volatility estimate from market ]

    Keyword Arguments:
        iters {int} -- [description] (default: {100})
    """
    for _ in range(iters):
        sigmaEst -= (bsmCallValue(St, K, t, T, r, sigmaEst) - C0)/bsmVega(St, K, t, T, r, sigmaEst)

    return sigmaEst
