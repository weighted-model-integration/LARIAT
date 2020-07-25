
from multiprocessing import Process
from numpy import asarray, isclose
from os import remove
import psutil
from pysmt.shortcuts import BOOL, get_env, REAL, serialize, And, read_smtlib, write_smtlib
from pywmi import Domain, XaddEngine
from pywmi.transform import normalize_formula
from random import choice
from string import ascii_uppercase, digits
from subprocess import CalledProcessError
from time import time
from wmilearn import logger
from wmilearn.exceptions import ModelException
from wmilearn.model import Model
from wmilearn.det import DET
from wmilearn.utils import check_Z_normalize

MSPN_import_err = None
try:
    from tfspn.SPN import SPN, Splitting
    from wmilearn.conversions import SPN_to_WMI

except ImportError as e:
    logger.warning("Couldn't import the MSPN library: " + str(e))    
    MSPN_import_err = e


def kill_recursive(pid):
    proc = psutil.Process(pid)
    for subproc in proc.children(recursive=True):
        try:
            subproc.kill()
        except psutil.NoSuchProcess:
            continue
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass


DEF_RENORM_TIMEOUT = 60 * 60

TEST_AND_NORM_SAMPLES = 1000000

# LARIAT modes
RENORM_OFF = 'off'
RENORM_BG_ONLY = 'bg'
RENORM_FULL = 'full'

TMP_CHARS = ascii_uppercase + digits
TMP_LEN = 10

class DETLearner:

    def __init__(self, learner_args):
        self.learner_args = learner_args
        self.det = None

    def estimate_density(self, training_data, validation_data=None):
        """Fit a DET on the training data. If the optional validation data is
        provided, it is used to prune the tree."""

        self.det = DET(**{k : v for k,v in self.learner_args.items()
                                      if k in ['n_min', 'n_max']})
        logger.info("Growing the full tree")
        self.det.grow_full_tree(training_data)
        logger.info(self.det.info())
        logger.info("Pruning the full tree")
        if validation_data is None:
            if 'n_bins' in self.learner_args:
                self.det.prune_with_cv(training_data, n_bins=self.learner_args['n_bins'])
            else:
                self.det.prune_with_cv(training_data)
        else:
            self.det.prune_with_validation(validation_data)

        logger.info(self.det.info())

    def renormalize(self, training_data, seed, mode=RENORM_OFF,
                    support=None, timeout=None, global_norm=False):

        if timeout is None:
            timeout = DEF_RENORM_TIMEOUT

        detcopy = self.det.copy()

        model_support = detcopy.tree_to_WMI_support()
        model_weight = detcopy.tree_to_WMI_weightfun()

        bounds = {v.symbol_name() : b for v, b in detcopy.root.bounds.items()
                  if v.symbol_type() == REAL}


        renorm_support = None
        if mode == RENORM_BG_ONLY and training_data.constraints is not None:
            renorm_support = training_data.constraints
        elif mode == RENORM_FULL:
            if training_data.constraints is not None and support is not None:
                renorm_support = training_data.constraints & support
            elif training_data.constraints is not None:
                renorm_support = training_data.constraints
            elif support is not None:
                renorm_support = support

        renormalized = False
        if renorm_support is not None:

            if global_norm:
                logger.debug("Global renormalization")
                model_support = model_support & renorm_support
                renormalized = True
            else:
                logger.debug("Local renormalization")
                def renorm_wrap(inst, support, support_path, weight_path):
                    try:
                        inst.renormalize(support)
                        support = inst.tree_to_WMI_support()
                        weight = inst.tree_to_WMI_weightfun()
                        msg = "Writing result to files:\n{}\n{}"
                        logger.debug(msg.format(support_path, weight_path))
                        write_smtlib(support, support_path)
                        write_smtlib(weight, weight_path)                    
                        logger.debug("Done.")
                    
                    except ModelException as e:
                        logger.error("Couldn't renormalize the DET: {}".format(e))

                # communication with wrapper process through file
                # NEVER use multiprocessing.Queue with huge pysmt formulas
                rndstr =  ''.join(choice(TMP_CHARS) for _ in range(TMP_LEN))
                support_path = "{}.support".format(rndstr)
                weight_path = "{}.weight".format(rndstr)
                timed_proc = Process(target=renorm_wrap, args=(detcopy,
                                                               renorm_support,
                                                               support_path,
                                                               weight_path))

                logger.debug("Starting renormalization with timeout: {}".format(timeout))
                timed_proc.start()
                logger.debug("Timed proc started")
                timed_proc.join(timeout)
                logger.debug("Timed proc joined")

                if timed_proc.is_alive():
                    logger.warning("Renormalization timed out")
                    pid = timed_proc.pid                
                    logger.warning("Killing process {} and its children".format(pid))
                    kill_recursive(pid)

                else:
                    try:
                        model_support = read_smtlib(support_path)
                        remove(support_path)
                    except FileNotFoundError:
                        model_support = None
                    try:
                        model_weight = read_smtlib(weight_path)
                        remove(weight_path)
                    except FileNotFoundError:
                        model_weight = None

                    if model_support is None or model_weight is None:
                        raise ModelException("Couldn't renormalize the DET")
                
                    logger.debug("Renormalization done")
                    renormalized = True

        model = Model(model_support,
                      model_weight,
                      list(map(lambda x : x[0], training_data.features)),
                      bounds,
                      metadata=self.learner_args)


        # is Z = 1?
        if renormalized:
            check_Z_normalize(model, seed, TEST_AND_NORM_SAMPLES)
            
        elif not global_norm:
            # fallback strategy for local: to global
            model, renormalized = self.renormalize(training_data, seed, mode=mode,
                                    support=support, timeout=timeout, global_norm=True)

        return model, renormalized

class MSPNLearner:

    def __init__(self, learner_args):
        if MSPN_import_err is not None:
            raise MSPN_import_err
        
        assert('seed' in learner_args)
        self.learner_args = learner_args
        self.spn = None

    def estimate_density(self, training_data, validation_data=None):
        """Fit a MSPN on the training data. The variable validation_data is
        never used."""
        feature_types = []
        feature_names = []
        families = []
        for feat, str_type in training_data.features:
            feature_types.append(str_type)
            feature_names.append(feat.symbol_name())
            if 'leaf' in self.learner_args:
                families.append(self.learner_args['leaf'])
            else:            
                families.append(MSPNLearner.SPN_feat_fams[feat.symbol_type()])

        if 'row_split' in self.learner_args:
            if self.learner_args['row_split'] == 'gower':
                row_split_method = Splitting.Gower(n_clusters=2)
            elif self.learner_args['row_split'] == 'rdc-kmeans':
                row_split_method = Splitting.KmeansRDCRows(n_clusters=2,
                                                           k=20,
                                                           OHE=1)
            else:
                raise NotImplementedError()

        else:
            row_split_method = Splitting.KmeansRDCRows(n_clusters=2,
                                                       k=20,
                                                       OHE=1)

        col_split_method = Splitting.RDCTest(threshold=0.1,
                                             OHE=1,
                                             linear=1)

        rand_seed = self.learner_args['seed']
        mspnargs = {k : v for k, v in self.learner_args.items()
                    if k not in ['seed', 'leaf', 'row_split']}

        # let MSPNs sort this out
        families = None
        self.spn = SPN.LearnStructure(asarray(training_data.data),
                                 feature_types,
                                 families=families,
                                 featureNames=feature_names,
                                 rand_seed=rand_seed,
                                 row_split_method=row_split_method,
                                 col_split_method=col_split_method,
                                 **mspnargs)


    def renormalize(self, training_data, seed, mode=RENORM_OFF,
                    support=None, timeout=None, global_norm=True):

        if timeout is None:
            timeout = DEF_RENORM_TIMEOUT

        feature_dict = {var.symbol_name() : var
                        for var, _ in training_data.features}

        model_weightfun, model_support = SPN_to_WMI(self.spn.root, feature_dict)

        bounds = {}
        for i, feat in enumerate(training_data.features):
            var = feat[0]
            if var.symbol_type() == REAL:
                xi = list(map(lambda row : row[i], training_data.data))
                bounds[var.symbol_name()] = [min(xi), max(xi)]

        renorm_support = None
        if mode == RENORM_BG_ONLY and training_data.constraints is not None:
            renorm_support = training_data.constraints
        elif mode == RENORM_FULL:
            if training_data.constraints is not None and support is not None:
                renorm_support = training_data.constraints & support
            elif training_data.constraints is not None:
                renorm_support = training_data.constraints
            elif support is not None:
                renorm_support = support

        renormalized = False
        if renorm_support is not None:
            if global_norm:
                logger.debug("Global renormalization")
                model_support = model_support & renorm_support
                renormalized = True

            else:
                logger.debug("Local renormalization")
                domain = Domain.make([v.symbol_name()
                                      for v, _ in training_data.features
                                      if v.symbol_type() == BOOL], bounds)

                nc_model_support = normalize_formula(model_support)
                nc_model_weightfun = normalize_formula(model_weightfun)
                nc_renorm_support = normalize_formula(renorm_support)

                t_0 = time()
                xaddsolver = XaddEngine(domain, nc_model_support, nc_model_weightfun,
                                        mode="original",
                                        timeout=timeout)

                t_init = time() - t_0
                logger.debug("XADDEngine t_init: {}".format(t_init))
                try:
                    res = xaddsolver.normalize(renorm_support)
                    t_norm = time() - t_init
                except CalledProcessError as e:
                    raise ModelException("CalledProcessError")

                if res is None:
                    logger.warning("Timeout.")
                else:
                    logger.debug("XADDEngine t_norm: {}".format(t_norm))
                    model_weightfun = get_env().formula_manager.normalize(res)
                    model_support = get_env().formula_manager.normalize(
                        And(model_support, renorm_support))
                    renormalized = True

        model = Model(model_support,
                      model_weightfun,
                      list(map(lambda x : x[0], training_data.features)),
                      bounds,
                      metadata=self.learner_args)

        if renormalized:
            check_Z_normalize(model, seed, TEST_AND_NORM_SAMPLES)

        elif not global_norm:
            # fallback strategy for local: to global
            model, renormalized = self.renormalize(training_data, seed, mode=mode,
                                    support=support, timeout=timeout, global_norm=True)            

        return model, renormalized
