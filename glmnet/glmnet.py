import numpy as np
import _glmnet

_DEFAULT_THRESHOLD = 1.0e-4
_DEFAULT_FLMIN = 0.001
_DEFAULT_NLAM = 100

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
            raise ValueError("Can't specify both lambdas and flmin keywords")
        ulam = np.asarray(kwargs['lambdas'])

        # Pass flmin > 1.0 indicating to use the user-supplied lambda values.
        flmin = 2.
        nlam = len(ulam)
    else:
        # If there are no user-provided lambdas, use flmin/nummodels to
        # specify the regularization levels tried.
        ulam = None
        flmin = kwargs['flmin'] if 'flmin' in kwargs else _DEFAULT_FLMIN
        nlam = kwargs['nummodels'] if 'nummodels' in kwargs else _DEFAULT_NLAM

    # Call the Fortran wrapper.
    lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr =  \
            _glmnet.elnet(balance, predictors, target, weights, jd, vp,
                          memlimit, flmin, ulam, thr, nlam=nlam)
    
    # Check for errors, documented in glmnet.f.
    if jerr != 0:
        if jerr == 10000:
            raise ValueError('cannot have max(vp) < 0.0')
        elif jerr == 7777:
            raise ValueError('all used predictors have 0 variance')
        elif jerr < 7777:
            raise MemoryError('elnet() returned error code %d' % jerr)
        else:
            raise Exception('unknown error: %d' % jerr)
    
    return GlmnetLinearResults(lmu, a0, ca, ia, nin, rsq, alm, nlp,
                               predictors.shape[1], balance)


class GlmnetLinearModel(object):
    def __init__(self, a0, ca, ia, nin, rsq, alm, npred):
        self._intercept = a0
        self._coefficients = ca[:nin]
        self._indices = ia[:nin] - 1
        self._rsquared = rsq
        self._lambda = alm
        self._npred = npred

    def __str__(self):
        return ("%s with %d non-zero coefficients (%.2f%%)\n" + \
                " * Intercept = %.7f, Lambda = %.7f\n" + \
                " * Training r^2: %.4f") % \
                (self.__class__.__name__, len(self._coefficients),
                 len(self._coefficients) / float(self._npred) * 100,
                 self._intercept, self._lambda, self._rsquared)

    def predict(self, predictors):
        predictors = np.atleast_2d(np.asarray(predictors))
        return self._intercept + \
                np.dot(predictors[:,self._indices], self._coefficients)
    
    @property
    def coefficients(self):
        coeffs = np.zeros(self._npred)
        coeffs[self._indices] = self._coefficients
        return coeffs



class GlmnetLinearResults(object):
    def __init__(self, lmu, a0, ca, ia, nin, rsq, alm, nlp, npred, parm):
        self._lmu = lmu
        self._a0 = a0
        self._ca = ca
        self._ia = ia
        self._nin = nin
        self._rsq = rsq
        self._alm = alm
        self._nlp = nlp
        self._npred = npred
        self._model_objects = {}
        self._parm = parm

    def __str__(self):
        ninp = np.argmax(self._nin)
        return ("%s object, elastic net parameter = %.3f\n" + \
                " * %d values of lambda\n" + \
            " * computed in %d passes over data\n" + \
            " * largest model: %d predictors (%.1f%%), train r^2 = %.4f") % \
            (self.__class__.__name__, self._parm, self._lmu, self._nlp, 
             self._nin[ninp], self._nin[ninp] / float(self._npred) * 100, 
             self._rsq[ninp]) 
    
    def __len__(self):
        return self._lmu

    def __getitem__(self, item):
        item = (item + self._lmu) if item < 0 else item
        if item >= self._lmu or item < 0:
            raise IndexError("model index out of bounds")

        if item not in self._model_objects:
            model =  GlmnetLinearModel(
                        self._a0[item],
                        self._ca[:,item],
                        self._ia,
                        self._nin[item],
                        self._rsq[item],
                        self._alm[item],
                        self._npred
                    )
            self._model_objects[item] = model
        
        else:
            model = self._model_objects[item]
        
        return model
    
    @property
    def nummodels(self):
        return self._lmu

    @property
    def coefficients(self):
        return self._ca[:np.max(self._nin), :self._lmu]

    @property
    def indices(self):
        return self._ia

def plot_paths(results, which_to_label=None):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.clf() 
    interactive_state = plt.isinteractive()
    for index, path in enumerate(results.coefficients):
        if which_to_label and results.indices[index] in which_to_label:
            if which_to_label[results.indices[index]] is None:
                label = "$x_{%d}$" % results.indices[index]
            else:
                label = which_to_label[results.indices[index]]
        else:
            label = None
        
            
        if which_to_label and label is None:
            plt.plot(path, ':')
        else:
            plt.plot(path, label=label)
    plt.xlim(0, results.nummodels - 1)
    if which_to_label is not None:
        plt.legend(loc='upper left')
    plt.title('Regularization paths')
    plt.xlabel('Model index (different values of $\\lambda$)')
    plt.ylabel('Value of regression coefficient $\hat{\\beta}_i$')
    
    plt.show()
    plt.interactive(interactive_state)
