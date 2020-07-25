
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
Y_LABEL = "Volume difference"

def average(lst):
    return sum(lst)/len(lst)

def parse_results(path, normalize_volumes):

    with open(path, 'rb') as f:
        results = pickle.load(f)

    # TODO: change, this is ugly
    tokens = basename(path).partition(".")[0].split("_")
    params = tuple(map(int, tokens[1:8]))
    method = "_".join(tokens[8:])

    n_bools = params[1]
    dres = {'vanilla' : [], 'incal' : [], 'thresh' : []}
    for result in results:

        vanilla_chi, vanilla_outside, vanilla_vol = result[0:3]
        incal_chi, incal_outside, _, incal_vol = result[3:7]
        thresh_chi, thresh_outside, _, thresh_vol = result[7:11]

        if (vanilla_vol is None or
            incal_vol is None or
            thresh_vol is None):
            print("Result discarded")
            continue

        if normalize_volumes:
            vanilla_vol /= (2 ** n_bools)
            incal_vol /= (2 ** n_bools)
            thresh_vol /= (2 ** n_bools)

        dres['vanilla'].append(vanilla_vol)
        dres['incal'].append(incal_vol)
        dres['thresh'].append(thresh_vol)        

    return params, method, dres
    

def read_results_folder(path, normalize_volumes):
    # this should filter the learned models
    result_files = sorted([f for f in listdir(path) if "learned" not in f
                           and "volume" in f])

    METHODS = ['vanilla', 'incal', 'thresh']

    lb_matrix = {m : dict() for m in METHODS}
    rh_matrix = {m : dict() for m in METHODS}
    
    for f in result_files:

        print("reading", f)
        
        res = parse_results(join(path,f), normalize_volumes)
        if res is None :
            continue

        params, method, results = res
        n_reals, n_bools, depth, n_train, n_valid, hyperplanes, literals = params

        for method in METHODS:
            
            if (n_reals, hyperplanes) not in rh_matrix[method]:
                rh_matrix[method][(n_reals, hyperplanes)] = []

            if (literals, n_bools) not in lb_matrix[method]:
                lb_matrix[method][(literals, n_bools)] = []

            rh_matrix[method][(n_reals, hyperplanes)].extend(results[method])
            lb_matrix[method][(literals, n_bools)].extend(results[method])
        
    return rh_matrix, lb_matrix
    
def plot(matrix, output_path, xlabel):

    label = {'vanilla' : "Trivial support",
             'incal' : "INCAL+ support",
             'thresh' : "Volumes around test-points"}

    linestyle = {'vanilla' : "-", 'incal' : "--", 'thresh' : "-."}
    marker = {'vanilla' : ".", 'incal' : "+", 'thresh' : "x"}
    #sorted_methods = ['vanilla', 'thresh', 'incal']
    sorted_methods = ['vanilla', 'incal']

    plt.figure(figsize=FIG_SIZE)        
    for nm, method in enumerate(sorted_methods):
        x_vals = sorted(list(matrix[method].keys()))
        y_vals = [average(matrix[method][x]) for x in x_vals]
        stdev = [sqrt(average([(y_vals[xindex] - y)**2 for y in matrix[method][x]]))
                 for xindex, x in enumerate(x_vals)]

        plt.plot(range(len(x_vals)), y_vals,
                 linewidth=LINEWIDTH,
                 color=COLORS[nm],
                 label=label[method],
                 linestyle=linestyle[method],
                 marker=marker[method])

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
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
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

    normalize_volumes = True
    args = parser.parse_args()
    rh_matrix, lb_matrix = read_results_folder(args.results_folder, normalize_volumes)

    if sum(len(rh_matrix[m]) for m in rh_matrix) > 3:
        plot(rh_matrix, args.output_path+"_rh", "(r, h)")

    if sum(len(lb_matrix[m]) for m in lb_matrix) > 3:
        plot(lb_matrix, args.output_path+"_lb", "(l, b)")
