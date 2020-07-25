
from numpy import asarray, mean
from os import listdir
from os.path import isfile, join
import pickle
from pysmt.environment import get_env, reset_env
from pysmt.shortcuts import read_smtlib, serialize, write_smtlib
from time import time

from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.learners import *
from wmilearn.supportlearner import learn_supports_adaptive
from wmilearn.exceptions import ModelException

# full MLC suite sorted by increasing number of features
EXPERIMENTS = ['balance-scale', 'iris', 'cars', 'diabetes', 'breast-cancer', 'glass2', 'glass',
               'breast', 'solar', 'cleve', 'heart', 'australian', 'crx', 'hepatitis', 'german',
               'german-org', 'auto', 'anneal-U']

FEAT_SUFFIX = ".features"
DATA_SUFFIX = ".{}.data"

SUPPORT_TEMPL = "{}_{}_{}.support"


def load_mlc_dataset(folder, name):

    feat_path = join(folder, name + FEAT_SUFFIX)
    features, constraints = Dataset.read_feats(feat_path)

    datasets = {}
    for split in ['train', 'valid', 'test']:
        data_path = join(folder, name + DATA_SUFFIX.format(split))
        data = Dataset.read_data(data_path, features)
        datasets[split] = Dataset(features, data, constraints)

    return datasets['train'], datasets['valid'], datasets['test']

def load_supports(output_folder, experiment_name):
    
    support_paths = [join(output_folder, f) for f in listdir(output_folder)
                if isfile(join(output_folder, f)) and f.endswith(".support")
                and experiment_name + "_" in f]

    supports = []
    if len(support_paths) > 0:
        logger.info("Found {} supports for {}".format(len(support_paths),
                                                      experiment_name))
        for path in support_paths:
            chi = read_smtlib(path)
            t_mult = float(path.split("_")[1])
            supports.append((chi, t_mult))

    return supports

def learn_and_dump_supports(output_folder, experiment_name, train, seed,
                            incal_timeout, negative_bootstrap):
    
    supports_considered = set()
    supports = []
    index = 0
    bg_knowledge = train.constraints
    for res in learn_supports_adaptive(train, seed, timeout=incal_timeout, bg_knowledge=bg_knowledge,
                                       negative_bootstrap=negative_bootstrap):

        chi, k, h, thresholds, t_mult = res
        chistr = serialize(chi)

        if chistr in supports_considered:
            continue
        
        supports_considered.add(chistr)
        path = join(output_folder, SUPPORT_TEMPL.format(experiment_name,
                                                            t_mult, index))
        write_smtlib(chi, path)
        chi = read_smtlib(path)
        supports.append((chi, t_mult))
        index += 1

    return supports


def learn_or_load_supports(output_folder, experiment_name, train, seed,
                           incal_timeout, negative_bootstrap):

    supports = load_supports(output_folder, experiment_name)

    if len(supports) == 0:
        supports = learn_and_dump_supports(output_folder, experiment_name,
                                           train, seed, incal_timeout,
                                           negative_bootstrap)

    if len(supports) == 0:
        # if no support is learned on the full space, restrict
        # the search in the numerical subspace
        logger.warning("No support learned on the full space. Projecting..")
        numerical_vars = [v for v,s in train.features
                          if s in ["continuous", "discrete"]]
        proj_train = train.project(numerical_vars)
        supports = learn_and_dump_supports(output_folder, experiment_name,
                                           proj_train, seed, incal_timeout,
                                           negative_bootstrap)

    return supports


def run_mlc(exp_folder, output_folder, method, learner, seed, incal_timeout,
            renorm_timeout, global_norm, negative_bootstrap, support_renorm=True):

    if method == 'mspn':
        global_norm = True

    results = {}
    for exp_name in EXPERIMENTS:

        results[exp_name] = dict()

        reset_env()
        get_env().enable_infix_notation = True

        train, valid, test = load_mlc_dataset(exp_folder, exp_name)

        train_valid = Dataset(train.features,
                              train.data + valid.data,
                              train.constraints)
        
        msg = "Loaded experiment '{}'. |train| = {} |valid| = {} |test| = {}"
        logger.info(msg.format(exp_name, len(train), len(valid), len(test)))

        t_0 = time()
        learner.estimate_density(train, validation_data=valid)
        t_f = time() - t_0
        logger.info("training time: {}".format(t_f))
        results[exp_name]['training_time'] = t_f

        logger.info("Vanilla model.")
        results[exp_name]['None'] = dict()

        try:
            t_0 = time()
            renorm_model, _ = learner.renormalize(train, seed, mode=RENORM_OFF, support=None,
                                                  global_norm=global_norm, timeout=renorm_timeout)
        
            t_f = time() - t_0
            results[exp_name]['None']['renorm_time'] = t_f
        
            valid_ll, valid_out = renorm_model.pointwise_log_likelihood(valid)
            logger.info("VALID LL: {}".format(valid_ll))
            logger.info("VALID OUT: {}".format(valid_out))        
            results[exp_name]['None']['valid-ll'] = valid_ll
            results[exp_name]['None']['valid-out'] = valid_out

            train_valid_ll, train_valid_out = renorm_model.pointwise_log_likelihood(train_valid)
            logger.info("TRAIN+VALID LL: {}".format(train_valid_ll))
            logger.info("TRAIN+VALID OUT: {}".format(train_valid_out))
            results[exp_name]['None']['train-valid-ll'] = train_valid_ll
            results[exp_name]['None']['train-valid-out'] = train_valid_out
        
            test_ll, test_out = renorm_model.pointwise_log_likelihood(test)
            logger.info("TEST LL: {}".format(test_ll))
            logger.info("TEST OUT: {}".format(test_out))
            results[exp_name]['None']['test-ll'] = test_ll
            results[exp_name]['None']['test-out'] = test_out
        except ModelException as e:
            logger.warning("Couldn't convert the vanilla model to WMI")

        logger.info("LARIAT-BG only.")
        results[exp_name]['bg'] = dict()

        try:
            t_0 = time()
            renorm_model, _ = learner.renormalize(train, seed, mode=RENORM_BG_ONLY, support=None,
                                                  global_norm=global_norm, timeout=renorm_timeout)
        
            t_f = time() - t_0
            results[exp_name]['bg']['renorm_time'] = t_f
        
            valid_ll, valid_out = renorm_model.pointwise_log_likelihood(valid)
            logger.info("VALID LL: {}".format(valid_ll))
            logger.info("VALID OUT: {}".format(valid_out))        
            results[exp_name]['bg']['valid-ll'] = valid_ll
            results[exp_name]['bg']['valid-out'] = valid_out

            train_valid_ll, train_valid_out = renorm_model.pointwise_log_likelihood(train_valid)
            logger.info("TRAIN+VALID LL: {}".format(train_valid_ll))
            logger.info("TRAIN+VALID OUT: {}".format(train_valid_out))
            results[exp_name]['bg']['train-valid-ll'] = train_valid_ll
            results[exp_name]['bg']['train-valid-out'] = train_valid_out
        
            test_ll, test_out = renorm_model.pointwise_log_likelihood(test)
            logger.info("TEST LL: {}".format(test_ll))
            logger.info("TEST OUT: {}".format(test_out))
            results[exp_name]['bg']['test-ll'] = test_ll
            results[exp_name]['bg']['test-out'] = test_out
        except ModelException as e:
            logger.warning("Couldn't renormalize with LARIAT-BG only")

        if support_renorm:
            supports = learn_or_load_supports(output_folder, exp_name, train_valid, seed,
                                              incal_timeout, negative_bootstrap)
        else:
            supports = []
            logger.warning("Skipping support learning")

        for chi, t_mult in supports:

            logger.info("LARIAT-INCAL+ -- tmult {}".format(t_mult))
            results[exp_name][t_mult] = dict()

            try:
                t_0 = time()
                renorm_model, renormd = learner.renormalize(train,
                                                            seed,
                                                            mode=RENORM_FULL,
                                                            support=chi,
                                                            timeout=renorm_timeout,
                                                            global_norm=global_norm)
                t_f = time() - t_0

                if not renormd:
                    logger.warning("Couldn't renormalize the model")
                    continue

            except ModelException as e:
                logger.warning("Model error: {}".format(e))
                continue

            results[exp_name][t_mult]['renorm_time'] = t_f
            valid_ll, valid_out = renorm_model.pointwise_log_likelihood(valid)
            logger.info("VALID LL: {}".format(valid_ll))
            logger.info("VALID OUT: {}".format(valid_out))
            results[exp_name][t_mult]['valid-ll'] = valid_ll
            results[exp_name][t_mult]['valid-out'] = valid_out

            train_valid_ll, train_valid_out = renorm_model.pointwise_log_likelihood(train_valid)
            logger.info("TRAIN+VALID LL: {}".format(train_valid_ll))
            logger.info("TRAIN+VALID OUT: {}".format(train_valid_out))
            results[exp_name][t_mult]['train-valid-ll'] = train_valid_ll
            results[exp_name][t_mult]['train-valid-out'] = train_valid_out
            
            test_ll, test_out = renorm_model.pointwise_log_likelihood(test)
            logger.info("TEST LL: {}".format(test_ll))
            logger.info("TEST OUT: {}".format(test_out))
            results[exp_name][t_mult]['test-ll'] = test_ll
            results[exp_name][t_mult]['test-out'] = test_out

        logger.info("Results on {}:\n{}".format(exp_name, results[exp_name]))

        output_filename = "{}_results".format(method)
        output_path = join(output_folder, output_filename)

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':

    from argparse import ArgumentParser
    from os.path import exists

    parser = ArgumentParser()
        
    parser.add_argument("-f", "--exp-folder", type=str, required=True,
                        help="Path to the MLC folder")

    parser.add_argument("-o", "--output-folder", type=str, required=True,
                        help="Path to the output folder")

    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Seed number")

    parser.add_argument("--incal-timeout", type=int, required=True,
                        help="Support learning timeout")    

    parser.add_argument("--renorm-timeout", type=int, required=True,
                        help="Renormalization timeout")

    parser.add_argument("--global-norm", action="store_true",
                        help="Use global instead of local normalization")

    helpneg = """Use negative bootstrap INCAL+ (def: 1.0), 
- an integer inticates absolute number of samples
- a float indicates the ratio wrt the training set size"""
    parser.add_argument("--negative-bootstrap", type=str, help=helpneg, default="1.0")


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

    run_mlc(args.exp_folder,
            args.output_folder,
            args.action,
            learner,            
            args.seed,
            args.incal_timeout,
            args.renorm_timeout,
            args.global_norm,
            args.negative_bootstrap)
