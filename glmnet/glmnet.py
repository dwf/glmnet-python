import numpy as np
import _glmnet

_DEFAULT_THRESHOLD = 1.0e-4
_DEFAULT_FLMIN = 0.001
_DEFAULT_NLAM = 100

# lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr = elnet(parm,x,y,w,jd,vp,nx,flmin,ulam,thr,[ka,ne,nlam,isd])

def _check_keywords(kwdict, name, default):
    if name in kwdict:
        return kwdict[name]
    else:
        return default

def elastic_net(predictors, target, balance, memlimit=None,
                largest=None, **kwargs):

    memlimit = predictors.shape[1] if memlimit is None else memlimit
    largest = predictors.shape[1] if largest is None else largest

    if memlimit < largest:
        raise ValueError('Need largest <= memlimit')

    elif balance < 0.0 or balance > 1.0:
        raise ValueError('Must have 0.0 <= balance <= 1.0')
    
    # Minimum change in largest predictor coefficient to continue processing
    thr = kwargs['threshold'] if 'threshold' in kwargs else _DEFAULT_THRESHOLD

    # Weights for the observation cases.
    weights = np.asarray(kwargs['weights']).copy() \
            if 'weights' in kwargs \
            else np.ones(predictors.shape[0])

    # Should we standardize the inputs?
    isd = bool(kwargs['standardize']) if 'standardize' in kwargs else True

    # Variable penalties for each predictor penalties[i] = 0 means don't
    # penalize predictor i.
    vp = np.asarray(kwargs['penalties']).copy() \
            if 'penalties' in kwargs \
            else np.ones(predictors.shape[1])
    
    # Predictors to exclude completely.
    if 'exclude' in kwargs:
        exclude = list(kwargs['exclude'])
        exclude = [len(exclude)] + exclude
        jd = np.array(exclude)
    else:
        jd = np.array([0])

    # Decide on regularization scheme based on keyword parameters.
    if 'lambdas' in kwargs:
        if 'flmin' in kwargs:
            raise ValueError("Can't specify both lambda and flmin keywords")
        ulam = np.asarray(kwargs['lambda'])
        # Pass flmin > 1.0 indicating to use the user-supplied lambda values.
        flmin = 2.
        nlam = len(ulam)
    else:
        ulam = None
        flmin = kwargs['flmin'] if 'flmin' in kwargs else _DEFAULT_FLMIN
        nlam = kwargs['nummodels'] if 'nummodels' in kwargs else _DEFAULT_NLAM

    return _glmnet.elnet(balance, predictors, target, weights, jd, vp,
                         memlimit, flmin, ulam, thr, nlam=nlam)
