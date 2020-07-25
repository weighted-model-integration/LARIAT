
from pysmt.shortcuts import serialize, BOOL, REAL, And, LE, Real
#from pywmi.exceptions import SamplingException

from incal.generator import Generator

from wmilearn.experiment import Experiment
from wmilearn import logger
from wmilearn.exceptions import ModelException
from wmilearn.model import Model
from wmilearn.problem import Problem
from wmilearn.utils import normalize, sample_dataset

from wmilearn.randommodels import ModelGenerator
    

def support_generator(how_many, b_count, r_count, bias, k, lits, h, sample_count,
                      ratio_percent, error_percent, seed):
    prefix = "random_support"
    ratio, errors = ratio_percent / 100, error_percent / 100
    producer = Generator(b_count, r_count, bias, k, lits, h, sample_count, ratio,
                         seed, prefix)
    supports = []
    while len(supports) < how_many:
        try:
            chi = producer.generate_formula().support
        except RuntimeError:
            logger.warning("Runtime error while sampling the support")
            continue

        supports.append(chi)

    return supports

def generate_experiment(seed, n_problems, n_train, n_valid, n_reals, n_bools,
                        depth, bias, k, literals, h, ratio, errors):

    logger.info("Generating experiment:\n" +
                "seed: {}\n".format(seed) +
                "n_problems: {}\n".format(n_problems) +
                "n_train: {}\n".format(n_train) +
                "n_valid: {}\n".format(n_valid) +
                "n_reals: {}\n".format(n_reals) +
                "n_bools: {}\n".format(n_bools) +
                "bias: {}\n".format(bias) +
                "k: {}\n".format(k) +
                "literals: {}\n".format(literals) +
                "h: {}\n".format(h) +
                "ratio: {}\n".format(ratio) +
                "errors: {}\n".format(errors))
                
    model_generator = ModelGenerator(n_reals, n_bools, seed,
                                     templ_bools="b{}",
                                     templ_reals="r{}",
                                     initial_bounds=[0, 1])

    problems = []
    while len(problems) < n_problems:
        try:
            # generating the ground truth model
            # not complex enough
            #chi = model_generator.generate_support_tree(depth)
            sample_count = 1000
            chi = support_generator(1, n_bools, n_reals, bias, k, literals, h,
                                    sample_count, ratio, errors, seed)[0]

            w = model_generator.generate_weights_tree(depth, nonnegative=True,
                                                      splits_only=True)

            boolean_vars = list(set(v for v in chi.get_free_variables()
                                    if v.symbol_type() == BOOL).union(
                                            set(model_generator.bools)))
            
            real_vars = list(set(v for v in chi.get_free_variables()
                                    if v.symbol_type() == REAL).union(
                                            set(model_generator.reals)))
            
            bounds = {v.symbol_name() : model_generator.initial_bounds
                      for v in real_vars}

            fbounds = And([And(LE(Real(bounds[var.symbol_name()][0]), var),
                               LE(var, Real(bounds[var.symbol_name()][1])))
                           for var in real_vars])
            model = Model(And(fbounds, chi), w, boolean_vars + real_vars, bounds)

            # use exact inference to normalize the ground truth
            sample_count = None
            normalize(model, seed, sample_count, engine='pa')

            logger.debug("model generator reals: {}".format(model_generator.reals))
            logger.debug("model generator IDs: {}".format(list(map(id, model_generator.reals))))

            logger.debug("model reals: {}".format(model.continuous_vars))
            logger.debug("model IDs: {}".format(list(map(id, model.continuous_vars))))

            # sampling the dataset from the ground truth model
            datasets = {}
            datasets['train'] = sample_dataset(model, n_train)
            datasets['valid'] = sample_dataset(model, n_valid)

        except ModelException as e:
            logger.debug(e.msg)
            continue
        
        logger.debug("Model {}\n".format(len(problems)+1) +
                     "chi: {}\n".format(serialize(model.support)) +
                     "w: {}\n".format(serialize(model.weightfun)))

        problem = Problem(model,
                          datasets,
                          bounds=bounds)

        problems.append(problem)

    # better safe than sorry?
    metadata = {'n_reals' : n_reals, 'n_bools' : n_bools, 'depth' : depth,
                'n_train' : n_train, 'n_valid' : n_valid, 'seed' : seed}
        

    return Experiment(problems, metadata=metadata)
        
    

if __name__ == '__main__':
    
    import argparse
    from os.path import exists

    parser = argparse.ArgumentParser()
        
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Experiment path")

    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Seed number")

    parser.add_argument("-n", "--n-problems", type=int, required=True,
                        help="Number of problem instances")

    parser.add_argument("-t", "--n-train", type=int, required=True,
                        help="Training set size for each instance")

    parser.add_argument("-v", "--n-valid", type=int, required=True,
                        help="Validation set size for each instance")

    parser.add_argument("-r", "--n-reals", type=int, required=True,
                        help="Number of continuous variables")

    parser.add_argument("-b", "--n-bools", type=int, required=True,
                        help="Number of Boolean variables")

    parser.add_argument("-d", "--depth", type=int, required=True,
                        help="Depth of the generated models")

    parser.add_argument("--bias", default="cnf")
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--literals", default=4, type=int)
    parser.add_argument("--hyperplanes", default=7, type=int)
    parser.add_argument("--ratio", default=90, type=int)
    parser.add_argument("--errors", default=0, type=int)

    args = parser.parse_args()

    # better check this first
    if exists(args.output_path):
        raise IOError("File exists: {}".format(args.output_path))

    experiment = generate_experiment(args.seed,
                                     args.n_problems,
                                     args.n_train,
                                     args.n_valid,
                                     args.n_reals,
                                     args.n_bools,
                                     args.depth,
                                     args.bias,
                                     args.k,
                                     args.literals,
                                     args.hyperplanes,
                                     args.ratio,
                                     args.errors)

    experiment.dump(args.output_path)
