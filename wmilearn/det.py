
from copy import copy, deepcopy
from math import fsum, log
import numpy as np

from pysmt.shortcuts import And, BOOL, LE, Ite, Not, REAL, Real, serialize

from pywmi import PredicateAbstractionEngine, Domain

from wmilearn import logger
from wmilearn.exceptions import ModelException


class Node:

    def __init__(self, parent, n_tot, n_min, n_max, train, bounds):
        assert(n_min < n_max), "NMIN >= NMAX"
        assert(len(train) >= n_min), "|DATA| < NMIN"

        self.parent = parent
        self.n_tot = n_tot
        self.bounds = bounds
        #self.volume = Node.compute_volume(self.bounds)
        self.n_node = len(train)
        self.data = train
        self.error = Node.compute_node_error(self.n_tot, self.n_node, self.bounds)
        #self.weight = None
        self.renorm_const = 1.0
        self.marked = False
        
        if self.n_node <= n_max:
            self.turn_into_leaf()
        else:
            tentative_split = self.split(n_min, train)

            if tentative_split is None:
                self.turn_into_leaf()

            else:
                best, pos_train, neg_train, pos_bounds, neg_bounds = tentative_split

                self.split_variable, self.split_value = best
                self.pos = Node(self, n_tot, n_min, n_max, pos_train, pos_bounds)
                self.neg = Node(self, n_tot, n_min, n_max, neg_train, neg_bounds)

        if not self.is_leaf():
            pw, pv = self.pos.weight, self.pos.volume
            nw, nv = self.neg.weight, self.neg.volume            
            assert(np.isclose(self.weight * self.volume,
                              (pw * pv) + (nw * nv))), "Invariant"
            assert(self.volume > self.neg.volume), "Volume should decrease"
            assert(self.volume > self.pos.volume), "Volume should decrease"
            

    def bounds_to_SMT(self):
        formula = []
        for var in self.bounds:
            if var.symbol_type() == REAL:
                lower, upper = self.bounds[var]
                formula.append(And(LE(Real(lower), var),
                                       LE(var, Real(upper))))

            elif self.bounds[var] is not None:
                bool_bound = var if self.bounds[var] else Not(var)
                formula.append(bool_bound)

        return And(formula)

    def compute_split_score(self, n_pos, n_neg, pos_bounds, neg_bounds):
        n_tot = self.n_tot

        return self.error \
            - Node.compute_node_error(n_tot, n_pos, pos_bounds) \
            - Node.compute_node_error(n_tot, n_neg, neg_bounds)


    def copy(self):
        copy_node = copy(self)
        if not copy_node.is_leaf():
            copy_node.pos = copy_node.pos.copy()
            copy_node.neg = copy_node.neg.copy()

        return copy_node


    def get_density(self, point):

        if self.is_outside_bounds(point):
            return 0.0

        elif self.is_leaf():
            return self.weight

        else:
            if self.split_variable.symbol_type() == REAL:
                if point[self.split_variable] <= self.split_value:
                    return self.pos.get_density(point)
                else:
                    return self.neg.get_density(point)
            else:
                if point[self.split_variable]:
                    return self.pos.get_density(point)
                else:
                    return self.neg.get_density(point)


    def get_leaves(self):
        if self.is_leaf():
            return [self]
        else:
            return self.pos.get_leaves() + self.neg.get_leaves()


    def merge_marked(self):
        assert(not self.marked), "Shouldn't be marked"
        if not self.is_leaf():
            assert(not(self.pos.marked and self.neg.marked)), "Children shouldn't be both marked"
            if self.pos.marked or self.neg.marked:
                self.turn_into_leaf()
            else:
                self.pos.merge_marked()
                self.neg.merge_marked()             

    def get_internal_nodes(self):
        if self.is_leaf():
            return []
        else:
            return [self] + self.pos.get_internal_nodes() + self.neg.get_internal_nodes()

    def is_boolean_split(self):
        return (not self.is_leaf()) and self.split_variable.symbol_type() == BOOL

    def is_continuous_split(self):
        return (not self.is_leaf()) and self.split_variable.symbol_type() == REAL

    def is_leaf(self):
        return (self.pos == None and self.neg == None)

    def is_outside_bounds(self, point):
        assert(len(point) == len(self.bounds)), "Dimension mismatch"

        for var, val in point.items():
            if var.symbol_type() == REAL:
                if val < self.bounds[var][0] or val > self.bounds[var][1]:
                    return True

            elif var.symbol_type() == BOOL:
                if self.bounds[var] is not None and self.bounds[var] != val:
                    return True

        return False

    def pretty_print(self):
        nodestr = "({} {} {} {})"
        if self.is_leaf():
            nodetype = "L {}".format(str(self.weight))
        else:
            varName = self.split_variable.symbol_name()
            if self.split_variable.symbol_type() == BOOL:
                condition = varName
            else:
                condition = "({} <= {})".format(varName,
                                                str(float(self.split_value)))

            nodetype = "C {}\n{}\n{}".format(condition, self.pos.pretty_print(),
                                           self.neg.pretty_print())
        return nodestr.format(self.bounds, self.weight, self.volume, nodetype)

    def renormalize_node(self, support):
        if self.is_leaf():

            domA = [var.symbol_name() for var in self.bounds if var.symbol_type() == BOOL]
            domX = []
            bs = []
            for var, b in self.bounds.items():
                if var.symbol_type() == REAL:
                    domX.append(var.symbol_name())
                    bs.append(tuple(b))
                    
            domain = Domain.make(domA, domX, bs)
            intersection = And(support, self.bounds_to_SMT())            
            engine = PredicateAbstractionEngine(domain, intersection, Real(1))
            intervol = engine.compute_volume()

            if not intervol > 0:
                raise ModelException("Non-positive leaf intersection volume")
                
            if self.volume != intervol:
                self.renorm_const = self.volume / intervol

            
        else:
            self.pos.renormalize_node(support)
            self.neg.renormalize_node(support)

    def split(self, n_min, train):
        assert(len(train.data) > 1), "Can't split a single instance"
        best_score = None

        for ncol, feat in enumerate(train.features):
            var = feat[0]

            if var.symbol_type() == REAL:
                # TODO: this doesn't need to be done at each iteration
                values = sorted(list({row[ncol] for row in train.data}))

                for i in range(len(values)-1):
                    split_val = (values[i]+values[i+1])/2.0
                    pos, neg = train.split(ncol, split_val)

                    if len(pos) < n_min or len(neg) < n_min:
                        continue

                    posB, negB = Node.compute_split_bounds(self.bounds,
                                                         var, split_val)
                    
                    score = self.compute_split_score(len(pos), len(neg),
                                                   posB, negB)
                    if best_score is None or best_score < score:
                        best_score = score
                        best_split = (var, split_val)
                        pos_train = pos
                        neg_train = neg
                        pos_bounds = posB
                        neg_bounds = negB

            elif var.symbol_type() == BOOL:
                split_val = True
                pos, neg = train.split(ncol)

                if len(pos) < n_min or len(neg) < n_min:
                    continue

                posB, negB = Node.compute_split_bounds(self.bounds, var, split_val)
                score = self.compute_split_score(len(pos), len(neg), posB, negB)
                if best_score is None or best_score < score:
                    best_score = score
                    best_split = (var, True)
                    pos_train = pos
                    neg_train = neg
                    pos_bounds = posB
                    neg_bounds = negB

            else:
                assert(False), "Unsupported variable type."

        if not best_score is None:
            return best_split, pos_train, neg_train, pos_bounds, neg_bounds
        

    def to_weight_function(self):
        if self.is_leaf():
            return Real(self.weight)            
        else:
            if self.split_variable.symbol_type() == BOOL:
                condition = self.split_variable
            else:
                condition = LE(self.split_variable, Real(self.split_value))

            return Ite(condition, self.pos.to_weight_function(),
                       self.neg.to_weight_function())

    def turn_into_leaf(self):
        self.pos = None
        self.neg = None

    @property
    def volume(self):
        return Node.compute_volume(self.bounds)

    @property
    def weight(self):
        return self.renorm_const * float(self.n_node) / (self.n_tot * self.volume)        
        

    @staticmethod
    def compute_node_error(n_tot, n_node, bounds):
        volume = Node.compute_volume(bounds)
        return -(pow(n_node,2) / (pow(n_tot,2) *volume))

    @staticmethod
    def compute_split_bounds(bounds, split_variable, split_value):
        pos_bounds, neg_bounds = dict(), dict()
        for var in bounds:
            varbounds = bounds[var]
            if var.symbol_type() == REAL:
                assert(varbounds is not None), "Continuous bounds can't be None"
                assert(len(varbounds) == 2), "Continuous bounds should have len 2"
                pos_bounds[var] = list(varbounds)
                neg_bounds[var] = list(varbounds)
            elif varbounds is not None:
                pos_bounds[var] = varbounds 
                neg_bounds[var] = varbounds
            else:
                pos_bounds[var] = None 
                neg_bounds[var] = None

        if split_variable.symbol_type() == REAL:
            pos_bounds[split_variable][1] = split_value
            neg_bounds[split_variable][0] = split_value
        else:
            assert(bounds[split_variable] is None), "Boolean split must be unassigned"
            pos_bounds[split_variable] = split_value
            neg_bounds[split_variable] = not split_value

        return pos_bounds, neg_bounds
    
    @staticmethod
    def compute_volume(bounds):
        assert(len(bounds) > 0), "Can't compute volume with no bounds"
        volume = 1
        for var in bounds:
            if var.symbol_type() == REAL:
                lower, upper = bounds[var]
                volume = volume * (upper - lower)
            else:
                if bounds[var] is None:
                    volume = volume*2

        assert(volume > 0), "Volume must be positive"
        return volume


class DET:

    RENORM_PRE = 'pre'
    RENORM_POST = 'post'

    DEF_EPSILON = 0.00001

    def __init__(self, n_min=5, n_max=10):
        self.n_min = n_min
        self.n_max = n_max
        self.root = None
        self.support = None

    def copy(self):
        copiedDET = DET(self.n_min, self.n_max)
        copiedDET.support = self.support
        copiedDET.root = self.root.copy()
        return copiedDET

    def grow_full_tree(self, train):
        self.N = len(train)
        initialBounds = DET.compute_initial_bounds(train)
        self.root = Node(None, self.N, self.n_min, self.n_max, train, initialBounds)

    def info(self):
        n_bsplits = len([n for n in self.root.get_internal_nodes()
                              if n.is_boolean_split()])
        n_csplits = len([n for n in self.root.get_internal_nodes()
                                 if n.is_continuous_split()])
        
        infostr = "BSplits: {} CSplits: {} Leaves: {} Wmin: {} Wmax: {} Wavg: {}"
        leaves = list(map(lambda x : x.weight, self.root.get_leaves()))
        wmin = min(leaves)
        wmax = max(leaves)
        wavg = sum(leaves)/len(leaves)
        return infostr.format(n_bsplits, n_csplits, len(leaves), wmin, wmax, wavg)

    def is_correct(self):
        raise NotImplementedError()

    def prune_with_cv(self, train, n_bins=10):
        # keep pruning the trees while possible
        # compute a finite set of alpha values
        trees = [(self.root, 0.0)]
        while not trees[-1][0].is_leaf():
            nextTree = trees[-1][0].copy()
            minAlpha = None
            for t in  nextTree.get_internal_nodes():
                alpha = DET.g(t)
                if minAlpha == None or alpha < minAlpha:
                   minAlpha = alpha
                   minT = t

            minT.turn_into_leaf()
            trees.append((nextTree, minAlpha))

        cvTrees = []
        cv_bins = train.get_CV_bins(n_bins)
        for i, cv_bin in enumerate(cv_bins):
            iTrain = cv_bin[0]
            iBounds = DET.compute_initial_bounds(iTrain)
            iTree = Node(None, len(iTrain), self.n_min, self.n_max, iTrain, iBounds)
            cvTrees.append(iTree)            

        regularization = [0.0 for _ in range(len(trees)-1)]
        for i, cvTree in enumerate(cvTrees):
            validation = cv_bins[i][1]
            alpha_cv_tree = cvTree.copy()
            
            for t in range(len(trees)-2):
                cvReg = 0.0
                for row in validation.data:
                    datapoint = {validation.features[j][0] : row[j]
                               for j in range(len(row))}
                    cvReg += alpha_cv_tree.get_density(datapoint)

                regularization[t] += 2.0 * cvReg / self.N

                est_alpha = 0.5 * (trees[t+1][1] + trees[t+2][1])
                DET.prune_tree(alpha_cv_tree, est_alpha)

            cvReg = 0.0                
            for row in validation.data:
                datapoint = {validation.features[j][0] : row[j]
                             for j in range(len(row))}
                cvReg += alpha_cv_tree.get_density(datapoint)

            regularization[len(trees)-2] += 2.0 * cvReg / self.N

        maxError = None
        for t in range(len(trees)-1):
            alpha_tree, alpha = trees[t]
            r_alpha_tree = DET.compute_tree_error(alpha_tree)
            error = regularization[t] + r_alpha_tree

            if maxError == None or error > maxError:
                maxError = error
                self.root = alpha_tree

    def prune_with_validation(self, validation, epsilon=None):

        if epsilon is None:
            epsilon = DET.DEF_EPSILON
        # keep pruning the trees while possible
        # compute a finite set of alpha values
        trees = [(self.root, 0.0)]
        while not trees[-1][0].is_leaf():
            nextTree = trees[-1][0].copy()
            minAlpha = None
            for t in  nextTree.get_internal_nodes():
                alpha = DET.g(t)
                if minAlpha == None or alpha < minAlpha:
                   minAlpha = alpha
                   minT = t

            minT.turn_into_leaf()
            trees.append((nextTree, minAlpha))

        max_likelihood = None
        for alpha_tree, alpha in trees:
            # compute log-likelihood of the validation set
            log_l = 0
            for row in validation.data:
                datapoint = {validation.features[j][0] : row[j]
                             for j in range(len(row))}
                density = alpha_tree.get_density(datapoint)
                log_l += log(density) if density else epsilon

            if max_likelihood is None or log_l > max_likelihood:
                max_likelihood = log_l
                self.root = alpha_tree

    def tree_to_WMI_support(self):
        formula = self.root.bounds_to_SMT()

        if self.support:            
            return And(formula, self.support)
        else:
            return formula

    def tree_to_WMI_weightfun(self):
        return self.root.to_weight_function()

    def renormalize(self, support):
        assert(support is not None), "Can't renormalize with support = None"
        self.support = support

        # mark the tree
        queue = []
        for leaf in self.root.get_leaves():
            domA = [var.symbol_name() for var in leaf.bounds if var.symbol_type() == BOOL]
            domX = []
            bs = []
            for var, b in leaf.bounds.items():
                if var.symbol_type() == REAL:
                    domX.append(var.symbol_name())
                    bs.append(tuple(b))
                    
            domain = Domain.make(domA, domX, bs)
            intersection = And(support, leaf.bounds_to_SMT())            
            engine = PredicateAbstractionEngine(domain, intersection, Real(1))
            intervol = engine.compute_volume()
            leaf.marked = intervol <= 0
            if leaf.marked:
                logger.debug("Marked a leaf")
                queue.append(leaf)

        while len(queue) > 0:
            n = queue.pop(0)
            if not n.parent.marked:
                if n.parent.pos.marked and n.parent.neg.marked:
                    n.parent.marked = True
                    queue.append(n.parent)

        self.root.merge_marked()        
        self.root.renormalize_node(support)


    @staticmethod
    def compute_initial_bounds(train):
        bounds = {}
        for ncol, feat in enumerate(train.features):
            var = feat[0]
            if var.symbol_type() == REAL:
                lower = None
                upper = None
                for row in train.data:
                    if lower is None or row[ncol] < lower:
                        lower = row[ncol]
                    if upper is None or row[ncol] > upper:
                        upper = row[ncol]

                bounds[var] = [lower, upper]

            elif var.symbol_type() == BOOL:
                vals = {row[ncol] for row in train.data}                
                if len(vals) == 2:
                    bounds[var] = None
                else:
                    assert(len(vals) == 1), "More than 2 Boolean values"
                    bounds[var] = list(vals)[0]

        return bounds

    @staticmethod
    def compute_tree_error(tree):
        return fsum([l.error for l in tree.get_leaves()])

    @staticmethod
    def g(node):
        return (node.error - DET.compute_tree_error(node))/(len(
            node.get_leaves()) - 1.0)
    
    @staticmethod
    def prune_tree(tree, alpha):
        for t in tree.get_internal_nodes():
            if DET.g(t) <= alpha:
                t.turn_into_leaf()
