
import glob
import shutil
import sys
import json
import pandas as pd
import argparse 
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import seaborn as sns
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--results-dir',
        default=os.path.realpath(os.path.join(dir_path, "..", "results")))
    parser.add_argument(
        '--out-dir',
        default=os.path.realpath(os.path.join(dir_path, "..", "plot-results")))
    parser.add_argument(
        '--dataset',
        default="all")

    args = parser.parse_args()
    return args

#EDITS ONLY MADE FOR LINEAR PLOTS AS OF NOW
def gen_plots(data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'augmentation_robustness')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']
    print(linear_data)

    #since catplot is a figure level function, it produces a new, separate plot which doesn't follow style of past graphs
    fig= sns.catplot(x='aug_type', y='result', hue='basetrain', data=linear_data, kind="point",linestyle="-", ci=None)
    plt.close(1) 
    plt.xlabel("Augmentation Sets")
    plt.ylabel("Accuracy")
    plt.title(options['data_name'].replace("_", " ").title())

    #when you run the program, it shows 2 graphs (sns graph and empty graph)
    #Figure 1 is the empty graph; Figure 2 is the sns graph
    # plt.show()

    #creates output file
    outplot = os.path.join(lin_dir, '{}.pdf'.format(options['data_name']))

    #saves Figure 1, but Figure 1 is the empty graph created (NOT SNS Graph)
    fig.savefig(outplot, format='pdf', bbox_inches='tight')




    #DIDNT MAKE CATPLOT FOR FINETUNE SINCE ALL DATA FOR AUGMENTATION WAS LINEAR (WILL EDIT PART THIS LATER)
def set_size(width, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def setup_plot(width=300):
    return plt.subplots(1, 1, figsize=set_size(width))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # setup plots
    sns.set_style('darkgrid')
    sns.set()

    frames = [] #array that collects dataframes from each file

    #gets all files that start with "resisc_" (still need a file with the baseline results)
    result_files = glob.glob(os.path.join(args.results_dir, "resisc_*.json"), recursive=True)
    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)
        data = pd.DataFrame(raw_data.values())

        aug_type = []
        num_rows = len(data)

        #adds an aug-type column, which is used for catplot in gen_plot
        if "crop_only" in resfile:
            aug_type = ["Crop only"] * num_rows
        elif "crop_blur" in resfile:
            aug_type = ["Crop + blur only"] * num_rows
        elif "rm_gray" in resfile:
            aug_type = ["Remove grayscale"] * num_rows
        elif "rm_color" in resfile:
            aug_type = ["Remove color"] * num_rows
        else:
            aug_type = ["Baseline"] * num_rows

        data["aug_type"] = aug_type
        print(data)
        print("*********************************")


        data = data.astype({
            "pretrain_iters": int
        })
        if data.result.max() > 1:
            data.result /= 100
        data.basetrain = data.basetrain.replace("imagenet_r50_supervised", "supervised imagenet init")
        data.basetrain = data.basetrain.replace("moco_v2_800ep", "moco imagenet init")
        data.basetrain = data.basetrain.replace("no", "random init")
        data.basetrain = data.basetrain.replace("none", "random init")    
        frames.append(data)

    #combines all the dataframes from each file into 1 large dataframe
    result = pd.concat(frames, ignore_index=True)

    #changes all the dataset names to resisc_augmentation
    result.dataset = "resisc_augmentation"
    dataname = result.dataset[0] 

    #prints new concatenated dataframe 
    print(result) 

    gen_plots(result, {
        'out_dir': args.out_dir,
        'data_name': dataname
    })

if __name__ == "__main__":
    main(parse_args())