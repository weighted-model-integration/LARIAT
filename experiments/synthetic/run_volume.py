
from argparse import ArgumentParser
from numpy import finfo, isclose
import pickle
from pysmt.shortcuts import *
from pywmi import evaluate_assignment
from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.experiment import Experiment
from wmilearn.model import Model
from wmilearn.utils import ISE
from wmilearn.supportlearner import learn_supports_adaptive

MAX_METRIC = finfo(float).max

def compute_volume(support, ground_truth, seed, n_samples):
    poly1 = Model(support, None, ground_truth.get_vars(),
                  ground_truth.bounds)
    poly2 = Model(Bool(False), Real(0), ground_truth.get_vars(),
                  ground_truth.bounds)
    return ISE(poly1, poly2, seed, n_samples, engine='rej')


def compute_volume_diff(support, ground_truth, seed, n_samples):
    poly1 = Model(support, None, ground_truth.get_vars(),
                  ground_truth.bounds)
    poly2 = Model(ground_truth.support, None, ground_truth.get_vars(),
                  ground_truth.bounds)
    return ISE(poly1, poly2, seed, n_samples, engine='rej')


def compute_outside(support, dataset):
    outside = 0
    for row in dataset.data:
        datapoint = {dataset.features[j][0].symbol_name() : row[j]
                     for j in range(len(row))}
        outside += (1 - evaluate_assignment(support, datapoint))

    return outside / len(dataset.data)

def compute_metric(support, ground_truth, dataset, thresholds, seed, n_samples):
    
    inside = 0
    for row in dataset.data:
        datapoint = {dataset.features[j][0].symbol_name() : row[j]
                     for j in range(len(row))}
        inside += evaluate_assignment(support, datapoint)


    boolean_indices = [i for i,f in enumerate(dataset.features)
                       if f[0].symbol_type() == BOOL]
    boolean_proj = lambda r : tuple(r[i] for i in boolean_indices)
    boolean_subspaces = set(boolean_proj(row) for row in dataset.data)
    subwise = []
    for sub_b in boolean_subspaces:
        pointwise = []
        for row in dataset.data:
            if boolean_proj(row) != sub_b:
                continue

            dimensionwise = []
            for i, feat in enumerate(dataset.features):
                var = feat[0]
                if var.symbol_type() == REAL:
                    th = thresholds[var.symbol_name()]
                    lb = row[i] - th
                    ub = row[i] + th
                    point = And(LE(Real(lb), var),
                                LE(var, Real(ub)))
                    dimensionwise.append(point)
                            
            pointwise.append(And(dimensionwise))

        assert(len(pointwise))
        muA = And([dataset.features[boolean_indices[i]][0]
                          if v else
                          Not(dataset.features[boolean_indices[i]][0])
                          for i,v in enumerate(sub_b)])
        subwise.append(And(muA, Or(pointwise)))

    Bthr = Or(subwise)
    vol_Bthr = compute_volume(Not(Bthr), ground_truth, seed, n_samples)
    if isclose(vol_Bthr, 0.0):
        return MAX_METRIC
    else:
        vol_support_Bthr = compute_volume(support & Not(Bthr), ground_truth, seed, n_samples)
        #return vol_support_Bthr / vol_Bthr + inside / len(dataset.data)
        outside = len(dataset.data) - inside
        return vol_support_Bthr / vol_Bthr + outside / len(dataset.data)
    
    
if __name__ == '__main__':


    parser = ArgumentParser()
        
    parser.add_argument("-e", "--experiment-path", type=str, required=True,
                        help="Path to the experiment file")

    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to the output file")

    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Seed number")
    
    parser.add_argument("-n", "--n-samples", type=int, required=True,
                        help="Number of samples for approximating IAE")    

    parser.add_argument("-t", "--timeout", type=int, help="Timeout")

    parser.add_argument("--negative-bootstrap", type=int,
                        help="How many negative samples use to bootstrap INCAL+ (def: 0)",
                        default=0)    
    
    args = parser.parse_args()    
    seed = args.seed
    n_samples = args.n_samples
        
    use_boolean_knowledge = True
    timeout = args.timeout if args.timeout else None

    logger.info("Running volume computations on {}".format(args.experiment_path))
    
    experiment = Experiment.read(args.experiment_path)

    results = []
    for i, problem in enumerate(experiment.problems):
        results.append([])
        logger.info("====================")
        logger.info("Problem {}".format(i))
        assert(problem.original_path is not None)

        train = problem.datasets['train']
        valid = problem.datasets['valid']
        #debug
        valid.data = valid.data[:2]
        
        gt = problem.model
        outside = lambda chi : compute_outside(chi, valid)
        metric = lambda chi, th: compute_metric(chi, gt, valid, th, seed, n_samples)
        volume = lambda chi : compute_volume_diff(chi, gt, seed, n_samples)

        bounds = []
        for i, feat in enumerate(train.features):
            var = feat[0]
            min_i = min(row[i] for row in train.data)
            max_i = max(row[i] for row in train.data)
            if var.symbol_type() == REAL:
                bounds.append(And(LE(Real(min_i), var),
                                  LE(var, Real(max_i))))
            elif use_boolean_knowledge and (min_i == max_i):
                bool_bound = var if min_i else Not(var)
                bounds.append(bool_bound)                

        trivial_support = And(bounds)
        trivial_outside = outside(trivial_support)
        trivial_vol = volume(trivial_support)
        logger.info("trivial_outside: {}".format(trivial_outside))
        logger.info("trivial_vol: {}".format(trivial_vol))
        results[-1].extend([
            trivial_support,
            trivial_outside,
            trivial_vol])

        best_incal = None
        best_threshold = None

        if len(problem.learned_supports) > 0:
            assert(len(problem.learned_supports) ==
                   len(problem.metadata['supports_metadata']))
            
            incal_supports = []
            for i, chi in enumerate(problem.learned_supports):
                metadata = problem.metadata['supports_metadata'][i]
                incal_supports.append((chi,
                                       metadata['support_k'],
                                       metadata['support_h'],
                                       metadata['support_thresholds'],
                                       metadata['support_threshold_mult']))

                
        else:
            incal_supports = learn_supports_adaptive(train, seed,
                                                     timeout=timeout,
                                                     prior=train.constraints,
                                                     negative_bootstrap=args.negative_bootstrap)
        supports_metadata = []
        for res in incal_supports:

            chi, k, h, thresholds, threshold_mult = res

            metadata = dict()
            metadata['support_k'] = k
            metadata['support_h'] = h
            metadata['support_seed'] = seed
            metadata['support_thresholds'] = thresholds
            metadata['support_threshold_mult'] = threshold_mult

            supports_metadata.append(metadata)
            
            chi_metric = metric(chi, thresholds)
            if best_incal is None or best_incal[0] > chi_metric:
                best_incal = chi_metric, chi

            boolean_indices = [i for i,f in enumerate(train.features)
                               if f[0].symbol_type() == BOOL]
            boolean_proj = lambda r : tuple(r[i] for i in boolean_indices)
            boolean_subspaces = set(boolean_proj(row) for row in train.data)
            subwise = []
            for sub_b in boolean_subspaces:
                pointwise = []
                for row in train.data:
                    if boolean_proj(row) != sub_b:
                        continue

                    dimensionwise = []
                    for i, feat in enumerate(train.features):
                        var = feat[0]
                        if var.symbol_type() == REAL:
                            th = thresholds[var.symbol_name()]
                            lb = row[i] - th
                            ub = row[i] + th
                            point = And(LE(Real(lb), var),
                                        LE(var, Real(ub)))
                            dimensionwise.append(point)
                            
                    pointwise.append(And(dimensionwise))

                assert(len(pointwise))
                muA = And([train.features[boolean_indices[i]][0]
                           if v else Not(train.features[boolean_indices[i]][0])
                           for i,v in enumerate(sub_b)])
                subwise.append(And(muA, Or(pointwise)))

            thr = Or(subwise)
            thr_metric = metric(thr, thresholds)
            if best_threshold is None or best_threshold[0] > thr_metric:
                best_threshold = thr_metric, thr

        if best_incal is not None:
            incal_metric, incal_support = best_incal
            incal_outside = outside(incal_support)
            incal_vol = volume(incal_support)
            logger.info("incal_outside: {}".format(incal_outside))
            logger.info("incal_metric: {}".format(incal_metric))
            logger.info("incal_vol: {}".format(incal_vol))
            results[-1].extend([incal_support,
                                incal_outside,
                                incal_metric,
                                incal_vol])
        else:
            results[-1].extend([None, None, None, None])

        if best_threshold is not None:
            threshold_metric, threshold_support = best_threshold
            threshold_outside = outside(threshold_support)
            threshold_vol = volume(threshold_support)
            logger.info("threshold_outside: {}".format(threshold_outside))
            logger.info("threshold_metric: {}".format(threshold_metric))        
            logger.info("threshold_vol: {}".format(threshold_vol))
            results[-1].extend([threshold_support,
                                threshold_outside,
                                threshold_metric,
                                threshold_vol])
        else:
            results[-1].extend([None, None, None, None])

        if len(incal_supports) > 0:
            problem.learned_supports = [x[0] for x in incal_supports]
            if problem.metadata is None:
                problem.metadata = dict()

            problem.metadata['supports_metadata'] = supports_metadata
            problem.update()

    with open(args.output_path, 'wb') as f:
        logger.info("Dumping {}".format(args.output_path))
        pickle.dump(results, f)
        
                
        
