
from os.path import abspath, basename, dirname, exists, join, relpath
import pickle

from pysmt.shortcuts import read_smtlib, write_smtlib
from wmilearn import logger
from wmilearn.dataset import Dataset
from wmilearn.model import Model


class Problem:

    MODEL_TEMPL = "{}.model"
    DATASET_TEMPL = "{}.{}_dataset"
    SUPPORT_TEMPL = "{}.learned_support_{}"

    def __init__(self, model, datasets, bounds=None, learned_supports=None,
                 metadata=None):
        self.model = model
        self.datasets = datasets
        self.bounds = bounds
        self.learned_supports = learned_supports or []
        self.original_path = None
        self.metadata = metadata

    def dump(self, problem_path):
        if problem_path is None and self.original_path is None:
            raise IOError("Unspecified path")
        elif problem_path is not None and exists(problem_path):
            raise IOError("File exists: {}".format(problem_path))
        elif problem_path is None and self.original_path is not None:
            msg = "Dumping the problem with no specified path, using {}"
            logger.debug(msg.format(self.original_path))
            problem_path = self.original_path

        problem_name = basename(problem_path)
        folder = abspath(dirname(problem_path))
        
        model_filename = Problem.MODEL_TEMPL.format(problem_name)
        model_path = join(folder, model_filename)

        if self.original_path is None:
            self.model.dump(model_path)

        index = {'model_path' : relpath(model_path, folder),
                 'dataset_paths' : {}}            

        for dataset_name, dataset in self.datasets.items():
            
            dataset_filename = Problem.DATASET_TEMPL.format(problem_name,
                                                            dataset_name)
            dataset_path = join(folder, dataset_filename)

            if self.original_path is None:
                dataset.dump(dataset_path)

            index['dataset_paths'][dataset_name] = relpath(dataset_path, folder)



        if len(self.learned_supports) > 0:
            
            index['support_paths'] = []
            
            for i, chi in enumerate(self.learned_supports):
                support_filename = Problem.SUPPORT_TEMPL.format(problem_name, i)
                support_path = join(folder, support_filename)
                logger.debug("Writing support file: {}".format(support_path))
                write_smtlib(chi, support_path)
                index['support_paths'].append(relpath(support_path, folder))

        if self.bounds is not None:
            index['bounds'] = self.bounds

        if self.metadata is not None:
            index['metadata'] = self.metadata

        with open(problem_path, 'wb') as f:
            pickle.dump(index, f)

    def update(self):
        assert(self.original_path is not None)
        self.dump(None)

    @staticmethod
    def read(path):
        with open(path, 'rb') as f:
            index = pickle.load(f)

        folder = abspath(dirname(path))
        model = Model.read(join(folder, index['model_path']))

        datasets = {}
        for dataset_name, rel_path in index['dataset_paths'].items():        
            datasets[dataset_name] = Dataset.read(join(folder, rel_path))
            
        bounds = bounds = index['bounds'] if 'bounds' in index else  None
        learned_supports = []
        metadata = None
        
        if 'support_paths' in index:
            for support_path in index['support_paths']:
                try:
                    learned_supports.append(read_smtlib(join(folder, support_path)))
                except FileNotFoundError as e:
                    logger.warning("Couldn't read: {}".format(path))

        if 'metadata' in index:
            metadata = index['metadata']
        
        problem = Problem(model, datasets, bounds=bounds,
                          learned_supports=learned_supports, metadata=metadata)

        problem.original_path = path
        return problem

