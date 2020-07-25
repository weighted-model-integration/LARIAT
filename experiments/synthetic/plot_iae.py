
from math import sqrt
import matplotlib.pyplot as plt
from os import listdir
from os.path import basename, join
import pickle
from wmilearn.model import Model

STYLE = 'ggplot'
plt.style.use(STYLE)
FIG_SIZE = (8, 5)
DPI = 200
LINEWIDTH = 1.8
#COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
COLORS = ['#E24A33', '#348ABD', '#8EBA42', '#FBC15E', '#FFB5B8', '#988ED5', '#777777']
Y_LABEL = "Approx. Integrated Absolute Error"

def average(lst):
    return sum(lst)/len(lst)

def parse_results(path):

    with open(path, 'rb') as f:
        results = pickle.load(f)

    # TODO: change, this is ugly
    tokens = basename(path).split("_")
    params = tuple(map(int, tokens[1:8]))
    method = "_".join(tokens[8:])

    #print("params", params)
    #print("method", method)

    if method == 'volume':
        return None

    avg_metric = []
    nosupport = []
    gt = []
    for result in results:
        if 'None' in result:
            nosupport.append(result['None']['approx-iae'])

        if 'gt-renorm' in result:
            gt.append(result['gt-renorm']['approx-iae'])

        if 'training_time' in result:
            training_time = result['training_time']

        if 'best' in result:
            avg_metric.append(result[result['best']]['approx-iae'])

    return params, method, avg_metric, nosupport, gt
    

def read_results_folder(path):
    # this should filter the learned models
    result_files = sorted([f for f in listdir(path) if "learned" not in f
                           and "volume" not in f])

    rh_matrix = dict()
    lb_matrix = dict()
    
    for f in result_files:

        print("reading", f)
        
        res = parse_results(join(path,f))
        if res is None :
            continue

        params, method, avg_metric, nosupport, gt = res
        n_reals, n_bools, depth, n_train, n_valid, hyperplanes, literals = params

        if "det" in method:
            methodstr = "DET"
        elif "mspn" in method:
            methodstr = "MSPN"
        else:
            print("unrecognized method", method)
            continue
            
        for label, l in [(methodstr, nosupport),
                         ("LARIAT-"+methodstr+" INCAL+", avg_metric),
                         ("LARIAT-"+methodstr+" GT", gt)]:
            if label not in rh_matrix:
                rh_matrix[label] = dict()
            if label not in lb_matrix:
                lb_matrix[label] = dict()
            
            if (n_reals, hyperplanes) not in rh_matrix[label]:
                rh_matrix[label][(n_reals, hyperplanes)] = []

            if (literals, n_bools) not in lb_matrix[label]:
                lb_matrix[label][(literals, n_bools)] = []

            rh_matrix[label][(n_reals, hyperplanes)].extend(l)
            lb_matrix[label][(literals, n_bools)].extend(l)
        
    return rh_matrix, lb_matrix
    
def plot(matrix, output_path, xlabel):

    sorted_methods = [['DET', 'LARIAT-DET INCAL+', 'LARIAT-DET GT'],
                      ['MSPN', 'LARIAT-MSPN INCAL+', 'LARIAT-MSPN GT']]
    linestyle = ["-", "--", "-."]
    marker = [".", "+", "x"]

    for methods in sorted_methods:
        plt.figure(figsize=FIG_SIZE)
        for nm, method in enumerate(methods):
            x_vals = sorted(list(matrix[method].keys()))
            y_vals = [average(matrix[method][x]) for x in x_vals]
            stdev = [sqrt(average([(y_vals[xindex] - y)**2 for y in matrix[method][x]]))
                     for xindex, x in enumerate(x_vals)]
            #sqrt(average([(v - expval)**2 for v in aggregate[t_mult]]))
            plt.plot(range(len(x_vals)), y_vals,
                     linewidth=LINEWIDTH,
                     color=COLORS[nm],
                     label=method,
                     linestyle=linestyle[nm],
                     marker=marker[nm])
            plt.fill_between(range(len(x_vals)),
                             [y_vals[i] - stdev[i] for i in range(len(x_vals))],
                             [y_vals[i] + stdev[i] for i in range(len(x_vals))],
                             color=COLORS[nm],                             
                             alpha=0.2)

        locs = list(range(len(x_vals)))
        ticks = list(map(str, x_vals))
        plt.xlabel(xlabel)
        plt.xticks(locs, ticks)
        plt.ylabel(Y_LABEL)
        plt.legend(loc="upper right")
        plt.savefig(output_path + "_" + methods[0],
                    dpi=DPI,
                    bbox_inches='tight', pad_inches=0)
        #plt.show()



if __name__ == '__main__':

    from argparse import ArgumentParser
    from os.path import exists

    from wmilearn.experiment import Experiment

    parser = ArgumentParser()
        
    parser.add_argument("-i", "--results-folder", type=str, required=True,
                        help="Path to the results folder")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to the output file")

    args = parser.parse_args()
    rh_matrix, lb_matrix = read_results_folder(args.results_folder)
    if sum((len(rh_matrix[m]) for m in rh_matrix)) > len(rh_matrix):
        plot(rh_matrix, args.output_path+"_rh", "(r, h)")
    if sum((len(lb_matrix[m]) for m in lb_matrix)) > len(lb_matrix):
        plot(lb_matrix, args.output_path+"_lb", "(l, b)")
