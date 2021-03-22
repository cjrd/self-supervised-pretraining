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
import matplotlib as mpl

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--results-dir',
        default=os.path.realpath(os.path.join(dir_path, "results_modified")))
    parser.add_argument(
        '--out-dir',
        default=os.path.realpath(os.path.join(dir_path, "plot-results")))
    parser.add_argument(
        '--dataset',
        default="all")
    parser.add_argument(
        '--basetrain',
        default="")

    args = parser.parse_args()
    return args

#only deals with linear plots 
def gen_plots(data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'aug_robust_no_supervised')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']
    linear_data = linear_data[linear_data.variant=="linear-eval-lr"]
    mpl.style.use('default')
    #since catplot is a figure level function, it produces a new, separate plot which doesn't follow style of past graphs
#     with sns.axes_style("white"):
    fig= sns.catplot(x='aug_type', y='result', hue='basetrain', data=linear_data, kind="point",s=10,
        linestyle="-", legend_out=False,order=["Baseline","Remove\ngrayscale","Remove\ncolor","Crop + blur\nonly","Crop\nonly"])
        
    #sets axis labels and title of graph
    fig.set(xlabel='Augmentation Sets', ylabel='Change of Accuracy from Baseline', title=options['data_name'].replace("_", " ").title())
    # fig._legend.set_title("Basetrain")
    fig.ax.legend(title="Basetrain", fontsize="large", title_fontsize='large')
    for ax in fig.axes.flat:
        ax.set_title(options['data_name'].replace("_", " ").title(), fontsize=20)
        ax.set_xlabel("Augmentation Sets", fontsize=18)

        if 'chexpert' in options['data_name']:
            ax.set_ylabel("AUROC Change", fontsize=18)
        else:
            ax.set_ylabel("Accuracy Change", fontsize=18)
        xticks = ["Baseline","Remove\ngrayscale","Remove\ncolor","Crop+Blur\nonly","Crop\nonly"]
        ax.set_xticklabels(xticks, rotation=0, fontsize=12)
        
        yticks = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels(yticks, rotation=0, fontsize=17)
        # ax.tick_params(axis='y', labelsize='large')

        ax.spines['bottom'].set_color('1')
        ax.spines['top'].set_color('1')
        ax.spines['right'].set_color('1')
        ax.spines['left'].set_color('1')
        ax.patch.set_facecolor('0.97')
        ax.grid(axis='y', color='grey', dashes=[10, 4])

#         print(ax.spines)

#         ax.legend(fontsize=8)
    #gets rid of other graph created 
    plt.close(1)

    #creates output file
    outplot = os.path.join(lin_dir, '{}_{}.pdf'.format(options['data_name']+ "_augmentation", "no_supervised"))

    #saves Figure 1, but Figure 1 is the empty graph created (NOT SNS Graph)
    fig.savefig(outplot, format='pdf', bbox_inches='tight')

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

def reduce(data):
    result = data['result'].tolist()

    best = 0
    for value in result:
        if best < float(value):
            best = float(value)

    data.result = best
    return data, best

def main(args):

    #must pass in dataset arg for proper results

    os.makedirs(args.out_dir, exist_ok=True)
    # setup plots
    sns.set_style('darkgrid')
    sns.set()

    frames = [] #array that collects dataframes from each file
    if(args.dataset == "all"):
        dataset_type = "*"
    else:
        dataset_type = args.dataset

    #gets all files that start with "resisc_" (still need a file with the baseline results)
    # result_files = glob.glob(os.path.join(args.results_dir, dataset_type + "*.json"), recursive=True)
    result_files1 = glob.glob(os.path.join(args.results_dir, dataset_type + "*crop*.json"), recursive=True)
    result_files2 = glob.glob(os.path.join(args.results_dir, dataset_type + "*color*.json"), recursive=True)
    result_files3 = glob.glob(os.path.join(args.results_dir, dataset_type + "*gray*.json"), recursive=True)
    result_files  = result_files1 + result_files2 + result_files3
    result_files.append(os.path.join(args.results_dir, dataset_type + "_results.json"))

    moco_baseline = imagenet_baseline = nobt_baseline = 0

    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)

        print(resfile)

        types = [] #used for concatenating the resultzs from each basetrained model
        data = pd.DataFrame(raw_data.values())


        #finds the relevant values for moco bt, imagenet supervised bt, and no bt
        #mainly done to work around the baseline json file (has extra info that we don't want)
        if (dataset_type + "_results.json") in resfile:
            linear_data = data[data.result_type=='linear-eval']
            linear_data = linear_data[linear_data.variant=="linear-eval-lr"]

            data_moco = linear_data[data.basetrain=="moco_v2_800ep"]
            data_moco = data_moco[data_moco.pretrain_iters=="5000"]
            data_moco = data_moco[data_moco.pretrain_data==dataset_type]
            data_moco, moco_baseline = reduce(data_moco)

            if args.basetrain == 'supervised':
                data_imagenet = linear_data[data.basetrain=="imagenet_r50_supervised"]
                data_imagenet = data_imagenet[data_imagenet.pretrain_iters=="50000"]
                data_imagenet = data_imagenet[data_imagenet.pretrain_data==dataset_type]
                data_imagenet, imagenet_baseline= reduce(data_imagenet)
                types.append(data_imagenet)

            data_nobt = linear_data[data.basetrain=="no"]
            data_nobt= data_nobt[data_nobt.pretrain_iters=="100000"]
            data_nobt= data_nobt[data_nobt.pretrain_data==dataset_type]
            data_nobt, nobt_baseline = reduce(data_nobt)


            #appended to list 
            types.append(data_moco)
            types.append(data_nobt)

            #all concat to new dataframe 
            data = pd.concat(types, ignore_index=True)

        #used to add new column called aug_type
        aug_type = []
        num_rows = len(data)

        #adds an aug-type column, which is used for catplot in gen_plot
        if "crop_only" in resfile:
            aug_type = ["Crop\nonly"] * num_rows
        elif "crop_blur" in resfile:
            aug_type = ["Crop + blur\nonly"] * num_rows
        elif "rm_gray" in resfile:
            aug_type = ["Remove\ngrayscale"] * num_rows
        elif "rm_color" in resfile:
            aug_type = ["Remove\ncolor"] * num_rows
        else:
            aug_type = ["Baseline"] * num_rows

        data["aug_type"] = aug_type


        #get baseline and update results

        data = data.astype({
            "pretrain_iters": int
        })


        data.basetrain = data.basetrain.replace("imagenet_r50_supervised", "supervised imagenet init")
        data.basetrain = data.basetrain.replace("moco_v2_800ep", "HPT")
        data.basetrain = data.basetrain.replace("no", "MoCo Random Init")
        data.basetrain = data.basetrain.replace("none", "MoCo Random Init")    
        frames.append(data)

    #combines all the dataframes from each file into 1 large dataframe
    result = pd.concat(frames, ignore_index=True)


    #changes all the dataset names to dataset_augmentation
    result.dataset = dataset_type# + "_augmentation"

    #prints new concatenated dataframe 
    print(result)
    print(not (result.result > 1).all())

    mult100 = False
    if not (result.result > 1).all():
        mult100 = True

    if args.basetrain == 'supervised':
        print(moco_baseline, imagenet_baseline, nobt_baseline)
    else:
        print(moco_baseline, nobt_baseline)



    result_moco = result[result.basetrain=="HPT"]
    result_moco.result = result_moco.result - moco_baseline

    result_nobt = result[result.basetrain=="MoCo Random Init"]
    result_nobt.result = result_nobt.result - nobt_baseline

    if args.basetrain == 'supervised':
        result_imagenet = result[result.basetrain=="supervised imagenet init"]
        result_imagenet.result = result_imagenet.result - imagenet_baseline
        result = pd.concat([result_moco, result_imagenet, result_nobt], ignore_index=True)
    else:
        result = pd.concat([result_moco, result_nobt], ignore_index=True)
    dataname = result.dataset[0] 

    if mult100:
        result.result *=100
    print(result.result)
    print(result)

    gen_plots(result, {
        'out_dir': args.out_dir,
        'data_name': dataname,
    })

if __name__ == "__main__":
    main(parse_args())
