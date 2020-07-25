
import numpy as np
import pickle
from os.path import abspath, basename, dirname, exists, join, relpath
from pysmt.shortcuts import BOOL, REAL, get_env, read_smtlib, write_smtlib, Real

from pywmi import Domain, evaluate, evaluate_assignment
from wmilearn import logger

LOG_ZERO = 1e-3

class Model:

    SUPPORT_TEMPL = "{}.support"
    WEIGHTF_TEMPL = "{}.weight"

    def __init__(self, support, weightfun, variables, bounds, metadata=None):
        self.support = support
        self.weightfun = weightfun or Real(1.0)
        self.bounds = bounds
        
        self.boolean_vars = [v for v in variables if v.symbol_type() == BOOL]
        self.continuous_vars = [v for v in variables if v.symbol_type() == REAL]

        self.domain = Domain.make(map(lambda v : v.symbol_name(), self.boolean_vars),
                                  {v.symbol_name() : list(bounds[v.symbol_name()])
                                   for v in self.continuous_vars})

        self.metadata = metadata

    def eval(self, data):
        # TODO: fix this
        raise NotImplementedError()
        data = np.array(data)
        result = np.zeros(data.shape)
        result[:] = LOG_ZERO       
        inside = evaluate(self.domain, self.support, data)
        result[inside] = np.log(evaluate(self.domain, self.weightfun, data[inside]))
        return result

    def log_likelihood(self, dataset):
            
        ll = 0.0
        outside = 0
        for row in dataset.data:
            datapoint = {dataset.features[j][0].symbol_name() : row[j]
                         for j in range(len(row))}
            px = self.get_density(datapoint)
            if px:
                res = np.log(px)
            else:
                res = LOG_ZERO
                outside += 1

            ll += res

        return ll/len(dataset), outside/len(dataset)

    def pointwise_log_likelihood(self, dataset):
            
        ll = []
        outside = []
        for row in dataset.data:
            datapoint = {dataset.features[j][0].symbol_name() : row[j]
                         for j in range(len(row))}
            px = self.get_density(datapoint)
            if px:
                ll.append(np.log(px))
                outside.append(False)
            else:
                ll.append(LOG_ZERO)
                outside.append(True)

        return ll, outside
        

    def into_support(self, point):
        return evaluate_assignment(self.support, point)

    def get_density(self, point):
        if not self.into_support(point):
            return 0.0
        else:
            return evaluate_assignment(self.weightfun, point)

    def dump(self, model_path):

        if exists(model_path):
            logger.warning("File exists: {}".format(model_path))

        model_name = basename(model_path)

        support_filename = Model.SUPPORT_TEMPL.format(model_name)
        weightf_filename = Model.WEIGHTF_TEMPL.format(model_name)

        folder = abspath(dirname(model_path))

        support_path = join(folder, support_filename)
        weightf_path = join(folder, weightf_filename)

        paths = [support_path, weightf_path]
        if any(exists(f) for f in paths):
            logger.warning("File(s) exist:\n"+"\n".join(paths))

        write_smtlib(self.support, support_path)
        write_smtlib(self.weightfun, weightf_path)

        varlist = [(v.symbol_name(), v.symbol_type()) for v in self.get_vars()]

        index = {'support_path' : relpath(support_path, folder),
                 'weightf_path' : relpath(weightf_path, folder),
                 'variables' : varlist,
                 'bounds' : self.bounds}

        if self.metadata is not None:
            index['metadata'] = self.metadata

        with open(model_path, 'wb') as f:
            pickle.dump(index, f)

    def get_vars(self):
        return self.boolean_vars + self.continuous_vars


    @staticmethod
    def read(path):
        with open(path, 'rb') as f:
            index = pickle.load(f)

        folder = abspath(dirname(path))
        support = read_smtlib(join(folder, index['support_path']))
        weightfun = read_smtlib(join(folder, index['weightf_path']))
        variables = [get_env().formula_manager.get_or_create_symbol(varname, vartype)
                     for varname, vartype in index['variables']]
        bounds = index['bounds']

        metadata = None
        if 'metadata' in index:
            metadata = index['metadata']
        
        return Model(support, weightfun, variables, bounds, metadata=metadata)
