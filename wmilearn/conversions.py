from numpy import isclose

from tfspn.tfspn import SumNode, ProductNode, BernoulliNode
from tfspn.tfspn import CategoricalNode
from tfspn.tfspn import PiecewiseLinearPDFNodeOld, PiecewiseLinearPDFNode, IsotonicUnimodalPDFNode, HistNode, KernelDensityEstimatorNode

from pysmt.shortcuts import And, BOOL, LE, Ite, Or, Plus, REAL, Real, Times

from wmilearn import logger

def datapoints_to_piecewise_linear(var, xs, ys):
    
    assert(len(xs) == len(ys)), "dimensions mismatch"
    assert(all(xs[i-1] < xs[i] for i in range(1,len(xs)))),\
        "x values should be sorted"

    else_branch = Real(0)
    for i in range(1, len(xs)):
        assert(xs[i-1] < xs[i])
        interval = And(LE(Real(float(xs[i-1])), var), LE(var, Real(float(xs[i]))))
        a = float((ys[i] - ys[i-1]) / (xs[i] - xs[i-1]))
        b = float(ys[i] - a * xs[i])
        poly = Plus(Times(Real(a), var), Real(b))

        else_branch = Ite(interval, poly, else_branch)

    return else_branch

def hist_to_piecewise_constant(var, breaks, ys):
    assert(len(breaks) == len(ys)), "dimensions mismatch"
    assert(all(breaks[i-1] < breaks[i] for i in range(1,len(breaks)))),\
        "bin bounds should be sorted"
        
    else_branch = Real(0)
    for i in range(1, len(breaks)):
        assert(breaks[i-1] < breaks[i])
        interval = And(LE(Real(breaks[i-1]), var), LE(var, Real(breaks[i])))
        else_branch = Ite(interval, Real(ys[i], else_branch))

    return else_branch

def boolean_leaf(var, p1, p2):
    assert(var.symbol_type() == BOOL)
    msg = "p1 + p2 != 1,\n {} + {} == {}"
    assert(isclose(p1 + p2, 1.0)), msg.format(p1, p2)
    
    return Ite(var, Real(float(p1)), Real(float(p2)))    

def SPN_to_WMI(node, feature_dict):
    
    w_children = []
    chi_children = []
    for child in  node.children:
        subw, subchi = SPN_to_WMI(child, feature_dict)
        w_children.append(subw)
        if subchi is not None:
            chi_children.append(subchi)

    if isinstance(node, SumNode):
        wmi_weights = list(map(lambda w : Real(float(w)), node.weights))
        weighted_sum = [Times(wmi_weights[i], w_children[i])
                        for i in range(len(wmi_weights))]
        w_node = Plus(weighted_sum)
        chi_node = Or(chi_children)

    elif isinstance(node, ProductNode):
        w_node = Times(w_children)
        chi_node = And(chi_children)

    else: # it's a leaf
        wmi_var = feature_dict[node.featureName]

        if isinstance(node, BernoulliNode):
            assert(0 <= node.p and node.p <= 1)
            w_node = boolean_leaf(wmi_var, node.p, 1-node.p)
            chi_node = None

        elif isinstance(node, CategoricalNode):
            # I think this is never going to be used
            assert(node.values == 2), "Not a Boolean variable"
            w_node = boolean_leaf(wmi_var, node.probs[0], node.probs[1])
            chi_node = None

        elif isinstance(node, PiecewiseLinearPDFNodeOld):
            # I think this is never going to be used            
            logger.debug("Var: {}".format(wmi_var.symbol_name()) 
                         + " x_range: {}".format(node.x_range)
                         + " y_range: {}".format(node.y_range)
                         + " dom: {}".format(node.domain))

            if wmi_var.symbol_type() == REAL:
                w_node = datapoints_to_piecewise_linear(wmi_var, node.x_range,
                                                        node.y_range)
                chi_node =  And(LE(Real(float(node.domain[0])), wmi_var),
                                LE(wmi_var, Real(float(node.domain[-1]))))
            else:
                w_node = boolean_leaf(wmi_var, node.y_range[2], node.y_range[1])
                chi_node = None

            logger.debug("Leaf: {}".format(w_node))

        elif isinstance(node, PiecewiseLinearPDFNode) or \
             isinstance(node, IsotonicUnimodalPDFNode):
            logger.debug("Var: {}".format(wmi_var.symbol_name()) 
                         + " x_range: {}".format(node.x_range)
                         + " y_range: {}".format(node.y_range)
                         + " dom: {}".format(node.domain))

            if wmi_var.symbol_type() == REAL:
                actual_prob = datapoints_to_piecewise_linear(wmi_var, node.x_range,
                                                             node.y_range)
                w_node = Plus(
                    Times(Real(float(1 - node.prior_weight)), actual_prob),
                    Times(Real(float(node.prior_weight)), Real(float(node.prior_density))))
                chi_node = And(LE(Real(float(node.domain[0])), wmi_var),
                               LE(wmi_var, Real(float(node.domain[-1]))))
            else:
                p_true = node.y_range[list(node.x_range).index(True)]
                p_false = node.y_range[list(node.x_range).index(False)]
                print("p_true", p_true, "p_false", p_false)
                w_node = boolean_leaf(wmi_var, p_true, p_false)

                """
                if isclose(p_true, 1.0):
                    chi_node = wmi_var
                elif isclose(p_true, 1.0):
                    chi_node = Not(wmi_var)
                else:
                    chi_node = None
                """
                chi_node = None
                
            logger.debug("Leaf: {}".format(w_node))

        elif isinstance(node, HistNode):
            actual_prob = hist_to_piecewise_constant(wmi_var, node.breaks,
                                                     node.densities)
            w_node = Plus(
                Times(Real(float(1 - node.prior_weight)), actual_prob),
                Times(Real(float(node.prior_weight)), Real(float(node.prior_density))))
            chi_node = And(LE(Real(float(node.domain[0])), wmi_var),
                           LE(wmi_var, Real(float(node.domain[-1]))))


        elif isinstance(node, KernelDensityEstimatorNode):
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Node type {} not supported".format(type(node)))

    return w_node, chi_node
        

    





