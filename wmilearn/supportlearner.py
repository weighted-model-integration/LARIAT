from typing import List, Tuple

from incal.learn import LearnOptions
from incal.violations.core import RandomViolationsStrategy
from incal.violations.virtual_data import OneClassStrategy
from incal.k_cnf_smt_learner import KCnfSmtLearner
from incal.parameter_free_learner import learn_bottom_up

from multiprocessing import Process, Queue
import numpy as np

from pysmt.shortcuts import BOOL, REAL, serialize

from pywmi.smt_print import pretty_print
from pywmi.domain import Domain

import random

from wmilearn import logger


DEF_TIMEOUT= 600 # INCAL+ timeout for a single call

DEF_INITIAL = 10.0
DEF_MULT = 10.0
DEF_HOPS = 10
DEF_MAX_MULT = 10 ** 5


class Distance(object):
    def __init__(self, domain, dist_f):
        # type: (Domain, Callable[[Domain, List[Union[float, bool]], List[Union[float, bool]]], float]) -> None
        self.domain = domain
        self.dist_f = dist_f

    def between(self, p1, p2):
        # type: (List[Union[float, bool]], List[Union[float, bool]]) -> float
        return self.dist_f(self.domain, p1, p2)

    def between_dicts(self, p1, p2):
        # type: (Dict[str: [float, bool]], Dict[str: [float, bool]]) -> float
        return self.dist_f(self.domain, [p1[v] for v in self.domain.variables], [p2[v] for v in self.domain.variables])

    @staticmethod
    def l_inf(domain, p1, p2):
        # type: (Domain, List[Union[float, bool]], List[Union[float, bool]]) -> float
        if all(p1[ib] == p2[ib] for ib, b in enumerate(domain.variables) if domain.var_types[b] == BOOL):
            return max(abs(p1[ir] - p2[ir]) / domain.domain_size(r)
                       for ir, r in enumerate(domain.variables)
                       if domain.var_types[r] == REAL)
        else:
            return 1

    @staticmethod
    def l1(domain, p1, p2):
        # type: (Domain, List[Union[float, bool]], List[Union[float, bool]]) -> float
        if all(p1[ib] == p2[ib] for ib, b in enumerate(domain.variables) if domain.var_types[b] == BOOL):
            return sum(abs(p1[ir] - p2[ir]) / domain.domain_size(r)
                       for ir, r in enumerate(domain.variables)
                       if domain.var_types[r] == REAL)
        else:
            return len(domain.real_vars)


def learn_supports_adaptive(dataset, seed, bg_knowledge=None, timeout=None, initial=None, mult=None,
                            hops=None, max_mult=None, negative_bootstrap=None):

    if timeout is None:
        timeout = DEF_TIMEOUT

    if initial is  None:
        initial = DEF_INITIAL

    if mult is None:
        mult = DEF_MULT

    if hops is None:
        hops = DEF_HOPS

    if max_mult is None:
        max_mult = DEF_MAX_MULT

    results = []
    discovered = set()
    t_mults = set()
    
    last = initial
    i = 0

    msg = "Adaptive support learning. timeout = {}, init = {}, mult = {}, hops = {}"
    logger.info(msg.format(timeout, initial, mult, hops))
    while i < hops and last < max_mult:
        logger.debug("i: {} last: {}".format(i, last))
        t_mults.add(last)
        res = learn_support(dataset, seed, last, timeout=timeout, bg_knowledge=bg_knowledge,
                            symmetry_breaking="mvn",
                            negative_bootstrap=negative_bootstrap)
        
        if res is not None:
            chi, k, h, thresholds = res
            chistr = serialize(chi)            
            smaller = {t for t in t_mults if t < last}
            
            if chistr not in discovered:
                discovered.add(chistr)
                results.append(res + (last,))

            if len(smaller) > 0:
                last = (last + max(smaller)) / 2
                i += 1
            else:
                last = last / mult

        else: # last t_mult timed out
            larger = {t for t in t_mults if t > last}
            if len(larger) > 0:
                last = (last + min(larger)) / 2
                i += 1
            else:
                last = last * mult

    return results
                

            

def learn_support(dataset, seed, threshold_mult, timeout=None, bg_knowledge=None,
                  symmetry_breaking=None,
                  negative_bootstrap=None):

    logger.info(f"Running INCAL+. Symmetry breaking = {symmetry_breaking} negative_bootstrap = {negative_bootstrap}")    

    # default might become symmetry_breaking = "mvn"
    if symmetry_breaking is None:
        symmetry_breaking = ""

    if negative_bootstrap is None:
        negative_bootstrap = 0
    else:
        try:
            # absolute count is specified with an integer
            negative_bootstrap = int(negative_bootstrap)
        except ValueError:
            pass

        # relative count (wrt |D|) is specified with a float
        negative_bootstrap = int(len(dataset) * float(negative_bootstrap))
            
    # compute bounds and add positive labels to the data
    bounds = {}
    for row in dataset.data:
        for i, feat in enumerate(dataset.features):
            var = feat[0]

            if var.symbol_type() == BOOL:
                continue

            varname = var.symbol_name()
            if not varname in bounds:
                bounds[varname] = [row[i], row[i]]
            else:
                if row[i] < bounds[varname][0]:
                    bounds[varname][0] = row[i]
                elif row[i] > bounds[varname][1]:
                    bounds[varname][1] = row[i]

    data = np.array(dataset.data)
    labels = np.ones(data.shape[0])

    # create a Domain instance
    varnames = []
    vartypes = {}
    for v, _ in dataset.features:
        varnames.append(v.symbol_name())
        vartypes[v.symbol_name()] = v.symbol_type()

    domain = Domain(varnames, vartypes, bounds)
    distance = Distance(domain, Distance.l_inf)

    max_closest = None
    for i1 in range(len(data)):
        min_distance = None
        for i2 in range(0, len(data)):
            if i1 != i2:
                p1, p2 = dataset.data[i1], dataset.data[i2]
                d = distance.between(p1, p2)
                min_distance = d if min_distance is None else min(min_distance, d)
        if min_distance < 1:
            max_closest = min_distance if max_closest is None else max(max_closest, min_distance)

    logger.debug("Maximum distance between closest neighbors: {}".format(max_closest))

    threshold = threshold_mult * max_closest
    logger.debug("Overall threshold: {}".format(threshold))

    thresholds = {r: threshold * domain.domain_size(r) for r in domain.real_vars}
    logger.debug("Thresholds per dimension: {}".format(thresholds))

    def learn_inc(_data, _labels, _i, _k, _h):
        strategy = OneClassStrategy(RandomViolationsStrategy(10), thresholds,
                                    background_knowledge=bg_knowledge)
        if negative_bootstrap > 0:
            _data, _labels = OneClassStrategy.add_negatives(domain, _data, _labels, thresholds, negative_bootstrap)

        learner = KCnfSmtLearner(_k, _h, strategy, symmetry_breaking)

        random.seed(seed)        
        initial_indices = LearnOptions.initial_random(20)(list(range(len(_data))))
        res = learner.learn(domain, _data, _labels, initial_indices)
        return res


    # wrapping INCAL+ into a timed process
    def learn_wrap(data, labels, learn_inc, queue):
        res = learn_bottom_up(data, labels, learn_inc, 1, 1, 1, 1, None, None)
        (new_data, new_labels, formula), k, h = res
        msg = "Learned CNF(k={}, h={})"
        logger.debug(msg.format(k, h))
        msg = "Data-set grew from {} to {} entries"
        logger.debug(msg.format(len(labels), len(new_labels)))
        
        queue.put((formula, k, h))

    queue = Queue()
    timed_proc = Process(target=learn_wrap, args=(data, labels, learn_inc, queue))
    timed_proc.start()
    timed_proc.join(timeout)
    if timed_proc.is_alive():
        # timed process didn't complete the job
        timed_proc.terminate()
        timed_proc.join()
        return None
    else:
        # get the learned formula, (k,h)
        chi, k, h = queue.get()
        return chi, k, h, thresholds


