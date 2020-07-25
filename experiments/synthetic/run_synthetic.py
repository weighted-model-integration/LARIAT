
import numpy as np
from os.path import abspath, basename, dirname, join
import pickle
from pysmt.shortcuts import Real, And, LE, REAL, serialize
from pywmi import Domain, evaluate
from subprocess import CalledProcessError
from time import time

from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.exceptions import ModelException
from wmilearn.learners import *
from wmilearn.model import Model
from wmilearn.utils import ISE, approx_IAE



DEF_TIMEOUT = 3600 # seconds


def run_problem(problem, learner, seed, n_samples, timeout,
                global_norm, use_lariat=True):

    ground_truth = problem.model
    evaluation = dict()

    train = problem.datasets['train']
    valid = problem.datasets['valid']

    train_valid = Dataset(train.features,
                          train.data + valid.data,
                          train.constraints)

    if problem.learned_supports is not None:
        prior_supports = {problem.metadata['supports_metadata'][i]
                          ['support_threshold_mult'] : chi
                          for i, chi in enumerate(problem.learned_supports)}
    else:
        logger.warning("Couldn't find any learned support.")
        prior_supports = dict()

    prior_supports['None'] = None
    prior_supports['gt-renorm'] = ground_truth.support

    t_0 = time()
    learner.estimate_density(train, validation_data=valid)
    t_f = time() - t_0
    logger.info("training time: {}".format(t_f))
    evaluation['training_time'] = t_f
    
    learned_models = []
    cached_models = dict()
    max_ll = None
    best = None

    logger.info("Evaluating:\n {}".format("\n".join(map(str,prior_supports.keys()))))
    
    for t_mult, prior_support in prior_supports.items():

        if t_mult != 'None' and not use_lariat:
            continue
        
        evaluation[t_mult] = dict()
        ps_str = serialize(prior_support) if not isinstance(t_mult, str) else t_mult
        
        if ps_str in cached_models:
            learned_model, evaluation[t_mult] = cached_models[ps_str]
        else:
            try:
                logger.info("--------------------------------------------------")
                logger.info("Support: {}".format(t_mult))

                mode = RENORM_FULL if prior_support is not None else RENORM_OFF
                t_0 = time()
                learned_model, renormd = learner.renormalize(train,
                                                             seed,
                                                             mode=mode,
                                                             support=prior_support,
                                                             timeout=timeout,
                                                             global_norm=global_norm)
                t_f = time() - t_0
                if not renormd and prior_support is not None:
                    continue

                evaluation[t_mult]['renorm_time'] = t_f

            except CalledProcessError as e:
                logger.warning("XADD error: {}".format(e))
                continue

            except ModelException as e:
                logger.warning("Model error: {}".format(e))
                continue

            logger.debug("Computing approx-IAE")
            iae = approx_IAE(learned_model, ground_truth, seed, n_samples)
            evaluation[t_mult]['approx-iae'] = iae

            logger.debug("Computing train-LL")
            train_ll, train_out = learned_model.log_likelihood(train)
            evaluation[t_mult]['train-ll'] = train_ll
            evaluation[t_mult]['train-out'] = train_out
            logger.debug("Computing valid-LL")
            valid_ll, valid_out = learned_model.log_likelihood(valid)
            evaluation[t_mult]['valid-ll'] = valid_ll
            evaluation[t_mult]['valid-out'] = valid_out
            train_valid_ll, train_valid_out = learned_model.log_likelihood(train_valid)
            evaluation[t_mult]['train-valid-ll'] = train_valid_ll
            evaluation[t_mult]['train-valid-out'] = train_valid_out

            if t_mult not in ['None','gt-renorm'] \
               and (max_ll is None or valid_ll > max_ll):
                max_ll = valid_ll
                best = t_mult

            logger.debug("Computing volume difference")
            poly1 = Model(learned_model.support, None, ground_truth.get_vars(),
                          ground_truth.bounds)
            poly2 = Model(ground_truth.support, None, ground_truth.get_vars(),
                       ground_truth.bounds)
            vol_diff = ISE(poly1, poly2, seed, n_samples, engine='rej')

            evaluation[t_mult]['vol-diff'] = vol_diff
            
            cached_models[ps_str] = (learned_model, evaluation[t_mult])

            domain = Domain.make(map(lambda v : v.symbol_name(), ground_truth.boolean_vars),
                                 learned_model.bounds)
            eval_falses = evaluate(domain, learned_model.support, np.asarray(train.data))

        learned_models.append((t_mult, learned_model))

    evaluation['best'] = best

    tmuls = sorted([key for key in evaluation
                    if key not in ['None', 'gt-renorm', 'training_time', 'best']])
                                   
    eval_msg = """RESULTS:
Training time: {}
No renorm: {}
GT renorm: {}
Best chi : {}

All chis:
{}
""".format(evaluation['training_time'], evaluation['None'],
           evaluation['gt-renorm'], (best, evaluation.get(best)),
           "\n".join([str((tmul, evaluation[tmul]))
                      for tmul in tmuls]))

    logger.info(eval_msg)

    return learned_models, evaluation


    
def run_experiment(experiment, learner, output_path, seed,
                   n_samples, global_norm, timeout=None,
                   discard_missing=True):

    if timeout is None:
        timeout = DEF_TIMEOUT
    
    results = []
    n_discarded = 0
    for i, problem in enumerate(experiment.problems):

        learned_models, evaluation = run_problem(problem, learner,
                                                 seed, n_samples,
                                                 timeout, global_norm)

        missing_data = (evaluation['gt-renorm'] == {} or
                        evaluation['None'] == {} or
                        evaluation['best'] is None)

        if discard_missing and missing_data:
            continue
            

        if learned_models is not None:
            j = 0
            for t_mult, learned_model in learned_models:
                output_name = basename(output_path)
                folder = abspath(dirname(output_path))
                model_name = output_name + "_{}_learned_{}".format(i, j)
                model_path = join(folder, model_name)
                learned_model.dump(model_path)
                evaluation[t_mult]['model_path'] = model_path
                j += 1

        results.append(evaluation)

    n_dis = len(experiment.problems) - len(results)
    n_tot = len(experiment.problems)
    logger.info("Experiment done. Discarded {}/{}.".format(n_dis, n_tot))

    if len(results) > 0:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        logger.warning("Nothing to dump!")



if __name__ == '__main__':

    from argparse import ArgumentParser
    from os.path import exists

    from wmilearn.experiment import Experiment

    parser = ArgumentParser()
        
    parser.add_argument("-e", "--experiment-path", type=str, required=True,
                        help="Path to the experiment file")

    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to the output file")

    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Seed number")

    parser.add_argument("-n", "--n-samples", type=int, required=True,
                        help="Number of samples for approximating IAE")

    parser.add_argument("--global-norm", action="store_true",
                        help="Use global instead of local normalization")

    parser.add_argument("-t", "--renorm-timeout", type=int,
                        help="Renormalization timeout in seconds")

    subparsers = parser.add_subparsers(dest="action")
    det_parser = subparsers.add_parser("det")
    mspn_parser = subparsers.add_parser("mspn")

    det_parser.add_argument("--n-min", type=int, help="Minimum leaf size")
    det_parser.add_argument("--n-max", type=int, help="Maximum leaf size")
    det_parser.add_argument("--n-bins", type=int, help="Number of bins used during CV")

    mspn_parser.add_argument("--min-inst-slice", type=int, help="min inst slice?")
    mspn_parser.add_argument("--alpha", type=float, help="alpha?")
    mspn_parser.add_argument("--prior-weight", type=float, help="prior weight?")
    mspn_parser.add_argument("--leaf", choices=['piecewise','isotonic'], help="leaf?")
    mspn_parser.add_argument("--row-split", choices=['rdc-kmeans','gower'], help="row split?")

    args = parser.parse_args()

    # better check this first
    if exists(args.output_path):
        logger.warning("File exists: {}".format(args.output_path))

    experiment = Experiment.read(args.experiment_path)

    learner_args = {}

    if args.seed :
        learner_args['seed'] = args.seed

    if args.action == 'det':

        if args.n_min:
            learner_args['n_min'] = args.n_min
        if args.n_max:
            learner_args['n_max'] = args.n_max
        if args.n_bins:
            learner_args['n_bins'] = args.n_bins

        learner = DETLearner(learner_args)

    elif args.action == 'mspn':
        if args.min_inst_slice:
            learner_args['min_instances_slice'] = args.min_inst_slice
        if args.alpha:
            learner_args['alpha'] = args.alpha
        if args.prior_weight:
            learner_args['prior_weight'] = args.prior_weight
        if args.leaf:
            learner_args['leaf'] = args.leaf
        if args.row_split:
            learner_args['row_split'] = args.row_split

        learner = MSPNLearner(learner_args)

    else:
        assert(False),"Unknown action"

    logger.info("Running {} on  experiment {}".format(args.action,
                                                      args.experiment_path))
    run_experiment(experiment, learner, args.output_path, args.seed,
                   args.n_samples, args.global_norm, timeout=args.renorm_timeout)
