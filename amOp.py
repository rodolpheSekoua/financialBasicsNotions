import numpy 
def generate_time_grid(time_step, maturity):
  return numpy.arange(0., maturity + time_step, time_step)


def laguerrePolynomial(LPrev,LPrevPrev, n, x):
    if n==0:
        return numpy.power(x,0)
    elif n==1 :
        return 1-x 
    return ((2*n-1 - x) * LPrev - (n-1) * LPrevPrev) / n

def laguerrePolynomialBasis(x):
    order = 4
    basis = []
    LPrev = None
    LPrevPrev = None
    for k in range(order+1):
        basis.append(laguerrePolynomial(LPrev,LPrevPrev, k, x))
        LPrevPrev = LPrev
        LPrev = basis[-1] 
    return numpy.vstack(basis).T

def polynomialRegressor(YTrain,xTrain,xEval,preTrained=False):
    order = 4
    #Fit the polynomial model with numpy.polyfit
    p = numpy.polyfit(xTrain,YTrain,order)
    #Evaluate the model for in the money paths, see numpy.polyval
    regressedValue = numpy.polyval(p,xEval)
    return regressedValue
    
def linearRegressor(YTrain,xTrain,xEval,preTrained=False):
    order = 4
    #Standard polynomial basis
    #regressionBasisTrain = numpy.vstack([numpy.power(xTrain,k) for k in range(order+1)]).T
    
    #Laguerre polynomial basis for in the money paths 
    regressionBasisTrain = laguerrePolynomialBasis(xTrain)
    #Solve the quadratic problem with numpy.linalg.lstsq function
    regressionCoefficient = numpy.linalg.lstsq(regressionBasisTrain, 
                                            YTrain, 
                                            rcond=None)[0]
    #Laguerre polynomial basis for all paths
    regressionBasisEval = laguerrePolynomialBasis(xEval)
    #Evaluate the linear model for all paths : sum coefficient time the basis function
    regressedValue = numpy.sum([regressionCoefficient[k] * regressionBasisEval[:,k] for k in range(order+1)],
                            axis=0)
    return regressedValue


def simulate(time_grid, time_step, nb_paths, b, sigma, S0):
  length = len(time_grid)
  
  q = 0 #dividend for curiosity
  mean = (b - q - 0.5 * sigma**2) * time_step
  std_dev = sigma * numpy.sqrt(time_step)
  
  #Same as previous exercise except that we simulate increments at different dates
  brownian_inc = numpy.random.normal(loc=0, scale=std_dev, size=(nb_paths, length))
  logReturn = brownian_inc * sigma + mean
  logReturn[:, 0] = 0.
  exp_gauss = numpy.exp(logReturn)
  
  return S0 * numpy.cumprod(exp_gauss, axis=1)[::-1]