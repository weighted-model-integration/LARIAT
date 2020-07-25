
from argparse import ArgumentParser

from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.experiment import Experiment
from wmilearn.supportlearner import learn_supports_adaptive
    
if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment-path", type=str, required=True,
                        help="Path to the experiment file")

    parser.add_argument("-s", "--seed", type=int, required=True,
                        help="Seed number")

    parser.add_argument("-t", "--timeout", type=int, help="Timeout")

    helpneg = """Use negative bootstrap INCAL+ (def: 1.0),
- an integer inticates absolute number of samples
- a float indicates the ratio wrt the training set size"""
    parser.add_argument("--negative-bootstrap", type=str, help=helpneg, default="1.0")

    args = parser.parse_args()
    timeout = args.timeout if args.timeout else None

    logger.info("Running support learning on {}".format(args.experiment_path))
    logger.info("Timeout: {}".format(timeout))
    logger.info("N. negatives: {}".format(args.negative_bootstrap))

    experiment = Experiment.read(args.experiment_path)

    for i, problem in enumerate(experiment.problems):
        logger.info("====================")
        logger.info("Problem {}".format(i))
        assert(problem.original_path is not None)

        if len(problem.learned_supports) > 0:
            msg = "Found {} supports. Skipping."
            logger.info(msg.format(len(problem.learned_supports)))
            continue

        learned_supports = []
        supports_metadata = []

        train = problem.datasets['train']
        valid = problem.datasets['valid']
        train_valid = Dataset(train.features,
                              train.data + valid.data,
                              train.constraints)

        for res in learn_supports_adaptive(train_valid, args.seed,
                                           timeout=timeout,
                                           bg_knowledge=train.constraints,
                                           negative_bootstrap=args.negative_bootstrap):

            chi, k, h, thresholds, threshold_mult = res
            learned_supports.append(chi)

            metadata = dict()
            metadata['support_k'] = k
            metadata['support_h'] = h
            metadata['support_seed'] = args.seed
            metadata['support_thresholds'] = thresholds
            metadata['support_threshold_mult'] = threshold_mult

            supports_metadata.append(metadata)

        if len(learned_supports) == 0:
            # try projecting on the continuous subspace
            logger.warning("No support learned on the full space. Projecting..")
            numerical_vars = [v for v,s in train_valid.features
                              if s in ["continuous", "discrete"]]
            projected_train_valid = train_valid.project(numerical_vars)

            for res in learn_supports_adaptive(projected_train_valid, args.seed,
                                           timeout=timeout,
                                           bg_knowledge=train.constraints,
                                           negative_bootstrap=args.negative_bootstrap):

                chi, k, h, thresholds, threshold_mult = res
                learned_supports.append(chi)

                metadata = dict()
                metadata['support_k'] = k
                metadata['support_h'] = h
                metadata['support_seed'] = args.seed
                metadata['support_thresholds'] = thresholds  
                metadata['support_threshold_mult'] = threshold_mult

                supports_metadata.append(metadata)
            

        if len(learned_supports) > 0:
            problem.learned_supports = learned_supports
            
            if problem.metadata is None:
                problem.metadata = dict()

            problem.metadata['supports_metadata'] = supports_metadata
            problem.update()

        else:
            logger.info("Couldn't learn any support")    
                
        
