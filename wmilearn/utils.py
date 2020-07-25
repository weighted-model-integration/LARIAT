
from copy import deepcopy
import numpy as np
from pysmt.shortcuts import And, BOOL, Bool, Ite, Minus, Not, Or, Plus, REAL, \
    Real, serialize, Times
from pywmi import Domain, RejectionEngine, PredicateAbstractionEngine, evaluate
from pywmi.sample import positive
from wmilearn.det import Node
from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.exceptions import ModelException
from wmilearn.model import Model


DEF_CLOSE_ENOUGH = 0.0001

def sample_dataset(model, n_samples):
    str_type = {BOOL : 'categorical', REAL : 'continuous'}
    features = []
    name_to_var = {}
    for var in model.get_vars():
        name_to_var[var.symbol_name()] = var
        features.append((var, str_type[var.symbol_type()]))

    features = [(var, str_type[var.symbol_type()]) for var in model.get_vars()]
    data = []

    samples, _ = positive(n_samples, model.domain, model.support, model.weightfun)
    for x in samples:
        row = [None for _ in range(len(x))]

        for index, varname in enumerate(model.domain.variables):
            var = name_to_var[varname]
            new_index = list(map(lambda x : x[0], features)).index(var)
            if var.symbol_type() == BOOL:
                row[new_index] = bool(x[index])
            else:
                row[new_index] = float(x[index])

        assert(not None in row)
        data.append(row)

    return Dataset(features, data, None)


def merged_domain(model1, model2):
    assert(model1.get_vars() == model2.get_vars())
    bounds = deepcopy(model1.bounds)
    for v, b in model2.bounds.items():
        if v not in bounds:
            bounds[v] = b
        else:
            bounds[v][0] = min(b[0], bounds[v][0])
            bounds[v][-1] = max(b[-1], bounds[v][-1])

    domain = Domain.make(map(lambda v : v.symbol_name(),
                             model1.boolean_vars), bounds)
    return domain, bounds
    


def normalize(model, seed, sample_count, engine='pa'):
    
    if engine == 'pa':
        solver = PredicateAbstractionEngine(model.domain, model.support, model.weightfun)
    elif engine == 'rej':
        solver = RejectionEngine(model.domain, model.support, model.weightfun,
                                 sample_count=sample_count, seed=seed)
    else:
        raise NotImplementedError()

    Z = solver.compute_volume()

    assert(Z >= 0), "Z is negative"

    if not np.isclose(Z, 1.0):
        logger.debug("Normalizing w with Z: {}".format(Z))
        model.weightfun = Times(Real(1.0/Z), model.weightfun)

    return Z

def check_Z_normalize(model, seed, sample_count):
    """Tests whether the model is normalized. If not, updates the weight
    function accordingly."""

    logger.debug("Approximating Z")
    solver = RejectionEngine(model.domain, model.support, model.weightfun,
                             sample_count=sample_count, seed=seed)
    all_ohes = dict()
    for var in model.domain.bool_vars:
        print("VAR:", var)
        if "_OHE_" in var:
            prefix = var.partition("_OHE_")[0]
            if prefix not in all_ohes:
                all_ohes[prefix] = []

            all_ohes[prefix].append(var)
    ohe_variables = list(all_ohes.values()) if len(all_ohes) > 0 else None
    Z_approx = solver.compute_volume(ohe_variables=ohe_variables)
    logger.debug("Z_approx: {}".format(Z_approx))
    if Z_approx <= 0:
        raise ModelException("Partition function is <= 0")
    
    if not abs(Z_approx - 1.0) <= DEF_CLOSE_ENOUGH:
        model.weightfun = Times(Real(float(1.0/Z_approx)), model.weightfun)


def approx_IAE(model1, model2, seed, sample_count):
    assert(set(model1.get_vars()) == set(model2.get_vars())),\
        "M1 vars: {}\n M2 vars: {}".format(model1.get_vars(),model2.get_vars())

    domain, bounds = merged_domain(model1, model2)

    samples, pos_ratio = positive(sample_count, domain,
                                  Or(model1.support, model2.support),
                                  weight=None)
    samples_m1 = samples[evaluate(domain,
                                  And(model1.support, Not(model2.support)),
                                  samples)]
    samples_m2 = samples[evaluate(domain,
                                  And(Not(model1.support), model2.support),
                                  samples)]
    samples_inter = samples[evaluate(domain, And(model1.support, model2.support),
                                  samples)]

    weights_m1 = sum(evaluate(domain, model1.weightfun, samples_m1))
    weights_m2 = sum(evaluate(domain, model2.weightfun, samples_m2))
    weights_inter = sum(abs(evaluate(domain, model1.weightfun, samples_inter) -
                        evaluate(domain, model2.weightfun, samples_inter)))

    n_m1 = len(samples_m1)
    n_m2 = len(samples_m2)
    n_inter = len(samples_inter)

    norm_m1 = weights_m1 / sample_count
    norm_m2 = weights_m2 / sample_count
    norm_inter = weights_inter / sample_count
    
    logger.debug(f"[ S1 ~S2] len: {n_m1}, sum: {weights_m1}, norm: {norm_m1}")
    logger.debug(f"[ S1 ~S2] len: {n_m2}, sum: {weights_m2}, norm: {norm_m2}")
    logger.debug(f"[ S1 ~S2] len: {n_inter}, sum: {weights_inter}, norm: {norm_inter}")

    approx_vol = pos_ratio * 2**len(domain.bool_vars)
    for lb, ub in bounds.values():
        approx_vol *= (ub - lb)

    return approx_vol*(weights_m1 + weights_m2 + weights_inter) / sample_count


def ISE(model1, model2, seed, sample_count, engine='pa'):

    assert(set(model1.get_vars()) == set(model2.get_vars())),\
        "M1 vars: {}\n M2 vars: {}".format(model1.get_vars(),model2.get_vars())
    
    support1, weightfun1 = model1.support, model1.weightfun
    support2, weightfun2 = model2.support, model2.weightfun


    support_d = Or(support1, support2)

    weight_d = Ite(And(support1, support2),
                   Times(Minus(weightfun1, weightfun2),
                         Minus(weightfun1, weightfun2)),
                   Ite(support1, Times(weightfun1, weightfun1),
                       Times(weightfun2, weightfun2)))

    domain, _ = merged_domain(model1, model2)

    if engine == 'pa':
        engine = PredicateAbstractionEngine(domain, support_d, weight_d)
        result = solver.compute_volume()

    elif engine == 'rej':
        result = None
        solver = RejectionEngine(domain, support_d, weight_d,
                                 sample_count=sample_count, seed=seed)        
        while result is None:
            #logger.debug("Attempting with sample_count {}".format(
                #solver.sample_count))
            result = solver.compute_volume()
            solver.sample_count *= 2
    else:
        raise NotImplementedError()    

    return result
