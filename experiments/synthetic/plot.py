
from math import sqrt
import matplotlib.pyplot as plt
from os import listdir
from os.path import basename, join
import pickle
from wmilearn.model import Model

label_metric = {'approx-ise' : "Approximate ISE",
                'approx-iae' : "Approximate IAE",
                'ise' : "Integrated Squared Error"}

def average(lst):
    return sum(lst)/len(lst)


def parse_volume_results(path, select_one=None):
    with open(path, 'rb') as f:
        results = pickle.load(f)

    # TODO: change, this is ugly
    tokens = basename(path).split("_")
    params = tuple(map(int, tokens[1:7]))

    print("params", params)

    if select_one is None:
        avg_diff = dict()
        xhklists = []
        for result in results:        
            assert('volumes' in result)
            xlist, y_list, hlist, klist = result['volumes']
            xhklists.append((xlist,hlist, klist))
            print("xlist", xlist)
            print("hlist", hlist)
            print("klist", klist)        
            for i, xi in enumerate(xlist):
                if xi not in avg_diff:
                    avg_diff[xi] = []
                avg_diff[xi].append(y_list[i])
            
        return params, avg_diff, xhklists

    else:
        vols = []
        for result in results:
            assert('volumes' in result)
            xlist, y_list, hlist, klist = result['volumes']
            try:
                i = xlist.index(select_one)
                vols.extend(y_list)
            except ValueError:
                continue

        return params, vols, None

def read_volume_results_folder(path, select_one=None):    
    # this should filter the learned models
    result_files = sorted([f for f in listdir(path) if "learned" not in f])
    aggregate = {}
    for f in result_files:        
        params, vols, xhklists = parse_volume_results(join(path,f), select_one)
        if select_one is None:
            for t_mult in vols:
                if t_mult not in aggregate:
                    aggregate[t_mult] = []

                aggregate[t_mult].extend(vols[t_mult])            
        else:
            n_reals, n_bools, depth, n_points, hyperplanes, literals = params
            if (hyperplanes, n_reals) not in aggregate:
                aggregate[(hyperplanes, n_reals)] = []
                
            aggregate[(hyperplanes, n_reals)].extend(vols)                

    x_vals = []
    y_vals = []
    for xval in sorted(list(aggregate.keys())):
        x_vals.append(xval)
        expval = average(aggregate[xval])
        stdev = sqrt(average([(v - expval)**2 for v in aggregate[xval]]))
        y_vals.append((expval, stdev))

    return x_vals, y_vals, xhklists

def plot_volume(results, output_path, select_one=None):
    x_vals, y_vals, xhklists = results
    expvals = list(map(lambda x : x[0], y_vals))
    stdevs =list(map(lambda x : x[1], y_vals))
    plt.plot(x_vals, expvals)
    plt.fill_between(x_vals,
                     [expvals[i] - stdevi for i, stdevi in enumerate(stdevs)],
                     [expvals[i] + stdevi for i, stdevi in enumerate(stdevs)],
                     alpha=0.2)

    if select_one is None:
        plt.xlabel("Threshold multiplier")
    else:
        plt.xlabel("(h, r) fixed threshold mult. {}".format(select_one))

    plt.ylabel("Volume difference wrt GT")
    plt.legend(loc="upper left")
    plt.savefig(output_path)
    plt.show()
    """
    i = 0
    for xlist, hlist, klist in xhklists:
        i += 1
        plt.plot(xlist, hlist, label="H{}".format(i))
        plt.plot(xlist, klist, label="K{}".format(i))

    plt.xlabel("Threshold multiplier")
    plt.ylabel("H, K")
    plt.legend(loc="upper right")
    plt.savefig(output_path+"_hk")
    plt.show()
    """

def parse_density_results(path, metric):
    def average(lst):
        return sum(lst)/len(lst)

    with open(path, 'rb') as f:
        results = pickle.load(f)

    # TODO: change, this is ugly
    tokens = basename(path).split("_")
    params = tuple(map(int, tokens[1:5]))
    method = "_".join(tokens[5:])

    print("params", params)
    print("method", method)

    if method == 'volume':
        return None

    avg_metric = dict()
    for result in results:
        for key in result:
            if key == 'None':
                nosupport = result['None'][metric]
            elif key == 'gt-renorm':
                gt = result['gt-renorm'][metric]
            elif key == 'training_time':
                training_time = result['training_time']
            else:
                t_mult = float(key)
                if t_mult not in avg_metric:
                    avg_metric[t_mult] = []
                if metric in result[t_mult]:
                    avg_metric[t_mult].append(result[t_mult][metric])

    return params, method, avg_metric, nosupport, gt


def read_density_results_folder(path, metric):
    # this should filter the learned models
    result_files = sorted([f for f in listdir(path) if "learned" not in f])
    
    res_matrix = {}
    nosupport_dict = {}
    gt_dict = {}    
    for f in result_files:
        
        res = parse_density_results(join(path,f), metric)
        if res is None :
            continue
        params, method, avg_metric, nosupport, gt = res
        n_reals, n_bools, depth, n_points = params
        
        if method not in nosupport_dict:
            nosupport_dict[method] = [nosupport]
        else:
            nosupport_dict[method].append(nosupport)

        if method not in gt_dict:
            gt_dict[method] = [gt]
        else:
            gt_dict[method].append(gt)  

        if method not in res_matrix:
            res_matrix[method] = dict()
            
        for t_mult in avg_metric:
            if t_mult not in res_matrix[method]:                
                res_matrix[method][t_mult] = avg_metric[t_mult]
            else:
                res_matrix[method][t_mult].extend(avg_metric[t_mult])
                
    approx_dict = {}
    for method in res_matrix:
        nosupport_dict[method] = average(nosupport_dict[method])
        gt_dict[method] = average(gt_dict[method])
        approx_dict[method] = dict()
        for t_mult in res_matrix[method]:
            approx_dict[method][t_mult] = average(res_matrix[method][t_mult])
            
    return approx_dict, nosupport_dict, gt_dict


def plot_density(results, output_path, metric):

    approx_dict, nosupport_dict, gt_dict = results

    for method in approx_dict:
        x_vals = sorted(list(approx_dict[method].keys()))
        y_vals = [approx_dict[method][t_mult] for t_mult in x_vals]
        gt = [gt_dict[method]] * len(x_vals)
        nosupport = [nosupport_dict[method]] * len(x_vals)
        plt.plot(x_vals, y_vals, label="support-"+method)
        plt.plot(x_vals, gt, label="gt-"+method)
        plt.plot(x_vals, nosupport, label="vanilla-"+method)        

    plt.xlabel("t_mult")
    plt.ylabel(label_metric[metric])
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    plt.show()

def parse_final_results(path, metric):
    def average(lst):
        return sum(lst)/len(lst)

    with open(path, 'rb') as f:
        results = pickle.load(f)

    # TODO: change, this is ugly
    tokens = basename(path).split("_")
    params = tuple(map(int, tokens[1:7]))
    method = "_".join(tokens[7:])

    print("params", params)
    print("method", method)

    if method == 'volume':
        return None

    avg_metric = []
    nosupport = []
    gt = []
    for result in results:
        for key in result:
            if key == 'None':
                nosupport.append(result['None'][metric])
            elif key == 'gt-renorm':
                gt.append(result['gt-renorm'][metric])
            elif key == 'training_time':
                training_time = result['training_time']
            elif metric in result[key]:
                avg_metric.append(result[key][metric])

    return params, method, avg_metric, nosupport, gt
    

def read_final_results_folder(path, metric):
    # this should filter the learned models
    result_files = sorted([f for f in listdir(path) if "learned" not in f])

    rh_matrix = dict()
    lb_matrix = dict()
    
    for f in result_files:

        print("reading", f)
        
        res = parse_final_results(join(path,f), metric)
        if res is None :
            continue

        params, method, avg_metric, nosupport, gt = res
        n_reals, n_bools, depth, n_points, hyperplanes, literals = params

        if method == 'volume':
            continue

        for variant in ["", "gt-", "vanilla-"]:
                        
            label = variant + method if method.startswith("det") else method
            if label not in rh_matrix:
                rh_matrix[label] = dict()
            if label not in lb_matrix:
                lb_matrix[label] = dict()
            
            if (n_reals, hyperplanes) not in rh_matrix[label]:
                rh_matrix[label][(n_reals, hyperplanes)] = []

            if (literals, n_bools) not in lb_matrix[label]:
                lb_matrix[label][(literals, n_bools)] = []

            if variant == "":
                tmp = avg_metric
            elif variant == "gt-":
                tmp = gt
            elif variant == "vanilla-":
                tmp = nosupport

            rh_matrix[label][(n_reals, hyperplanes)].extend(tmp)
            lb_matrix[label][(literals, n_bools)].extend(tmp)
            

    for key in rh_matrix:
        print(key, rh_matrix[key])

        
    return rh_matrix, lb_matrix
    
def plot_final(results, output_path, metric):

    rh_matrix, lb_matrix = results

    for method in rh_matrix:
        print("METHOD:", method)
        x_vals = sorted(list(rh_matrix[method].keys()))
        y_vals = [average(rh_matrix[method][rh]) for rh in x_vals]
        stdev = [sqrt(average([(y_vals[rhindex] - y)**2 for y in rh_matrix[method][rh]]))
                      for rhindex, rh in enumerate(x_vals)]
            #sqrt(average([(v - expval)**2 for v in aggregate[t_mult]]))
        plt.plot(range(len(x_vals)), y_vals, label=method)
        plt.fill_between(range(len(x_vals)),
                     [y_vals[i] - stdev[i] for i in range(len(x_vals))],
                     [y_vals[i] + stdev[i] for i in range(len(x_vals))],
                     alpha=0.2)
        

    plt.xlabel("(r, h)")
    locs, ticks = plt.xticks()
    plt.xticks(locs, map(str, x_vals))    
    plt.ylabel(label_metric[metric])
    plt.legend(loc="upper right")
    plt.savefig(output_path+"_rh")
    plt.show()

    for method in lb_matrix:
        x_vals = sorted(list(lb_matrix[method].keys()))
        y_vals = [average(lb_matrix[method][lb]) for lb in x_vals]
        stdev = [sqrt(average([(y_vals[lbindex] - y)**2 for y in lb_matrix[method][lb]]))
                      for lbindex, lb in enumerate(x_vals)]
            #sqrt(average([(v - expval)**2 for v in aggregate[t_mult]]))
        plt.plot(range(len(x_vals)), y_vals, label=method)
        plt.fill_between(range(len(x_vals)),
                     [y_vals[i] - stdev[i] for i in range(len(x_vals))],
                     [y_vals[i] + stdev[i] for i in range(len(x_vals))],
                     alpha=0.2)
        

    plt.xlabel("(l, b)")
    locs, ticks = plt.xticks()
    plt.xticks(locs, map(str, x_vals))
    plt.ylabel(label_metric[metric])
    plt.legend(loc="upper right")
    plt.savefig(output_path+"_lb")
    plt.show()

if __name__ == '__main__':

    from argparse import ArgumentParser
    from os.path import exists

    from wmilearn.experiment import Experiment

    parser = ArgumentParser()
        
    parser.add_argument("-i", "--results-folder", type=str, required=True,
                        help="Path to the results folder")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to the output file")

    subparsers = parser.add_subparsers(dest="action")
    volume_parser = subparsers.add_parser("volume")
    volume_parser.add_argument("-t", "--t-mult", type=float,
                        help="Fixed threshold multiplier")    
    density_parser = subparsers.add_parser("density")
    density_parser.add_argument("-m", "--metric", type=str, required=True,
                        help="Evaluation metric")
    final_parser = subparsers.add_parser("final")
    final_parser.add_argument("-m", "--metric", type=str, required=True,
                        help="Evaluation metric")

    args = parser.parse_args()
    if args.action == 'volume':
        results = read_volume_results_folder(args.results_folder,
                                             select_one=args.t_mult)
        plot_volume(results, args.output_path, select_one=args.t_mult)        
    elif args.action == 'density':
        results = read_density_results_folder(args.results_folder, args.metric)
        plot_density(results, args.output_path, args.metric)
    elif args.action == 'final':
        results = read_final_results_folder(args.results_folder, args.metric)
        plot_final(results, args.output_path, args.metric)
