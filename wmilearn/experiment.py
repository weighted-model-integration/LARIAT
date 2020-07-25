
from os.path import abspath, basename, dirname, exists, join, relpath
import pickle

from wmilearn import logger
from wmilearn.problem import Problem

class Experiment:

    PROBLEM_TEMPL = "{}.problem_{}"

    def __init__(self, problems, metadata=None):
        self.problems = problems
        self.metadata = metadata

    def dump(self, experiment_path):

        if exists(experiment_path):
            logger.warning("File exists: {}".format(experiment_path))

        experiment_name = basename(experiment_path)
        folder = abspath(dirname(experiment_path))

        problem_paths = []
        for i, prob in enumerate(self.problems):
            problem_filename = Experiment.PROBLEM_TEMPL.format(experiment_name, i)
            problem_path = join(folder, problem_filename)
            prob.dump(problem_path)
            problem_paths.append(relpath(problem_path, folder))

        index = {'problem_paths' : problem_paths}

        if self.metadata is not None:
            index['metadata'] = self.metadata

        with open(experiment_path, 'wb') as f:
            pickle.dump(index, f)

        return experiment_path
            

    @staticmethod
    def read(path):
        with open(path, 'rb') as f:
            index = pickle.load(f)

        metadata = None
        problems = []
        folder = abspath(dirname(path))
        for problem_path in index['problem_paths']:
            problems.append(Problem.read(join(folder,problem_path)))

        if 'metadata' in index:
            metadata = index['metadata']

        return Experiment(problems, metadata=metadata)
