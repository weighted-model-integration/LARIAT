
from math import ceil
from os.path import abspath, basename, dirname, exists, join, relpath
import pickle
from string import ascii_letters
from pysmt.shortcuts import BOOL, REAL, And, Bool, Not, Or, Symbol, read_smtlib,\
    write_smtlib

from wmilearn import logger

def mutex(variables):
    formulas = [Or(variables)]
    for i in range(len(variables)-1):
        for j in range(i+1, len(variables)):
            formulas.append(Not(And(variables[i], variables[j])))

    return And(formulas)


class Dataset:

    FEATS_TEMPL = "{}.features" 
    DATA_TEMPL = "{}.data"
    CONSTRAINTS_TEMPL = "{}.constraints"    
    FEATCHARS = ascii_letters + '0123456789'
    OHE_ID = "_OHE_"

    def __init__(self, features, data, constraints):
        self.data = data
        self.features = features
        self.constraints = constraints

    def __len__(self):
        return len(self.data)

    def split(self, ncol, val=None):
        pos_data = []
        neg_data = []

        if val == None:
            # if no split value is passed, the feature type must be boolean
            assert(self.features[ncol][0].symbol_type() == BOOL)
            test = lambda x : x
        else:
            test = lambda x : x <= val

        for row in self.data:
            if test(row[ncol]):
                pos_data.append(row)
            else:
                neg_data.append(row)

        pos_dataset = Dataset(list(self.features), pos_data, self.constraints)
        neg_dataset = Dataset(list(self.features), neg_data, self.constraints)
        return pos_dataset, neg_dataset

    def get_CV_bins(self, n_bins):
        assert(n_bins > 0 and n_bins <= len(self))
        bins = []
        bin_size = int(ceil(len(self) / float(n_bins)))
        for i in range(n_bins):
            imin = i * bin_size
            imax = (i+1) * bin_size
            train = Dataset(self.features, self.data[:imin]+self.data[imax:],
                            self.constraints)
            valid = Dataset(self.features, self.data[imin:imax],
                            self.constraints)

            assert(len(train)+len(valid)==len(self))
            bins.append((train, valid))

        return bins

    def dump(self, dataset_path):

        if exists(dataset_path):
            logger.warning("File exists: {}".format(dataset_path))

        dataset_name = basename(dataset_path)

        feats_filename = Dataset.FEATS_TEMPL.format(dataset_name)
        data_filename = Dataset.DATA_TEMPL.format(dataset_name)
        constr_filename = Dataset.CONSTRAINTS_TEMPL.format(dataset_name)

        folder = abspath(dirname(dataset_path))

        feats_path = join(folder, feats_filename)
        data_path = join(folder, data_filename)
        constr_path = join(folder, constr_filename)
        
        self.dump_feats(feats_path)
        self.dump_data(data_path)

        index = {'feats_path' : relpath(feats_path, folder),
                 'data_path' : relpath(data_path, folder)}

        if self.constraints is not None:
            write_smtlib(self.constraints, constr_path)
            index['constr_path'] = relpath(constr_path, folder)

        with open(dataset_path, 'wb') as f:
            pickle.dump(index, f)

    def project(self, variables):
        proj_feats = [f for f in self.features if f[0] in variables]
        indices = [self.features.index(f) for f in proj_feats]
        proj_data = [[row[v] for v in indices] for row in self.data]
        proj_constraints = None # TODO?

        return Dataset(proj_feats, proj_data, proj_constraints)
            

    @staticmethod
    def read(path):
        with open(path, 'rb') as f:
            index = pickle.load(f)

        folder = abspath(dirname(path))

        feats, constr = Dataset.read_feats(join(folder, index['feats_path']))
        data = Dataset.read_data(join(folder, index['data_path']), feats)

        if 'constr_path' in index:
            constr2 = read_smtlib(index['constr_path'])
            if constr is None:
                constr = constr2
            else:
                constr = And(constr, constr2)

        return Dataset(feats, data, constr)

    @staticmethod
    def read_feats(feats_path):
        features = []
        constraints = []
        with open(feats_path, 'r') as f:
            for line in f:
                if len(line) == 0:
                    continue

                tokens = line.strip()[:-1].split(':')
                assert(len(tokens) > 0), "Couldn't parse any token"

                feat_name = "".join(c for c in tokens[0] if c in Dataset.FEATCHARS)
                str_type = tokens[1]
                
                if len(tokens) > 2:
                    # actually we don't care about these
                    vals = tokens[2].split(",")
                
                if str_type == "continuous":
                    feat_type = REAL
                    features.append((Symbol(feat_name, feat_type), str_type))
                elif str_type == "discrete":
                    #feat_type = INT
                    feat_type = REAL
                    features.append((Symbol(feat_name, feat_type), str_type))
                elif str_type == "categorical":
                    feat_type =  BOOL
                    if len(vals) == 2:
                        features.append((Symbol(feat_name, feat_type), str_type))
                    else:
                        mutex_vars = []
                        for i in range(len(vals)):
                            ohe_feat_name = feat_name + Dataset.OHE_ID + str(i)
                            mutex_var = Symbol(ohe_feat_name, feat_type)
                            features.append((mutex_var, str_type))
                            mutex_vars.append(mutex_var)

                        constraints.append(mutex(mutex_vars))

        return features, And(constraints) if len(constraints) > 0 else None

    """
    @staticmethod
    def read_data(data_path, features):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if len(line) == 0:
                    continue

                tokens = line[:-1].replace(' ','').split(',')
                assert(len(tokens) == len(features))
                row = []
                for i,token in enumerate(tokens):
                    if features[i][0].symbol_type() == BOOL:
                        assert(token in ["0", "1"])
                        row.append(bool(int(token)))
                    else:
                        row.append(float(token))

                data.append(row)

        return data
    """

    @staticmethod
    def read_data(data_path, features):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if len(line) == 0:
                    continue

                tokens = line[:-1].replace(' ','').split(',')                
                row = []
                i = 0
                for token in tokens:
                    if Dataset.OHE_ID not in features[i][0].symbol_name():
                        if features[i][0].symbol_type() == BOOL:
                            assert(token in ["0", "1"])
                            row.append(bool(int(token)))
                        else:
                            row.append(float(token))

                        i += 1

                    else:
                        prefix = "_".join(
                            features[i][0].symbol_name().split("_")[:-1])
                        cardinality = len([f for f,_ in features if prefix in
                                           f.symbol_name()])
                        assert(int(float(token)) < cardinality)
                        subrow = [False for _ in range(cardinality)]
                        subrow[int(float(token))] = True
                        row.extend(subrow)
                        i += cardinality

                data.append(row)

        return data

    def dump_feats(self, feats_path):
        if exists(feats_path):
            logger.warning("File exists: {}".format(feats_path))
        
        with open(feats_path, 'w') as f:
            for feature, str_type in self.features:
                # TODO?: MSPNs dataset have a list of continuous values that is not
                # written here                
                if feature.symbol_type() == REAL:
                    assert(str_type in ['continuous', 'discrete'])
                    f.write("{}:{}.\n".format(feature.symbol_name(), str_type))
                else:
                    assert(str_type == 'categorical')
                    f.write("{}:categorical:0,1.\n".format(feature.symbol_name()))

                
    def dump_data(self, data_path):
        if exists(data_path):
            logger.warning("File exists: {}".format(data_path))

        with open(data_path, 'w') as f:
            for row in self.data:
                str_row = ",".join(map(str, row))\
                            .replace("True","1")\
                            .replace("False","0") + "\n"
                f.write(str_row)


if __name__ == '__main__':
    from sys import argv
    FEAT_SUFFIX = ".features"
    DATA_SUFFIX = ".{}.data"
    folder = "/home/morettin/code/MSPN/mlutils/data/MLC/proc-db/proc/cat/"
    name = argv[1]
    feat_path = join(folder, name + FEAT_SUFFIX)
    features, constraints = Dataset.read_feats(feat_path)

    split = 'test'
    data_path = join(folder, name + DATA_SUFFIX.format(split))
    data = Dataset.read_data(data_path, features)
    dataset = Dataset(features, data, constraints)

    print(dataset.features)
    print()
    print(dataset.data[0:5])
    
