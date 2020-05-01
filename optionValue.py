import numpy


def callValue(K, *S):
    """Value of call options

    Arguments:
        K {[float]} -- [strike price]
        S {[tuple]} -- [span of stock (valueStart, valueEnd, nbPts)]
    """
    span = numpy.linspace(*S)
    call = numpy.maximum(span - K, 0)
    return call

def putValue(K,*S):
    """Value of put optins

    Arguments:
        K {[float]} -- [strike price]
        S {[float]} -- [span of stock (valueStart, valueEnd, nbPts)]
    """
    span = numpy.linspace(*S)
    put  = numpy.maximum(K - span, 0)
    return put
