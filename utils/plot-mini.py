
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
# sns.set_style("whitegrid")


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

def logit_y_axis(yfact=100):
    # logit y axes with readable labels
    plt.yscale('logit')
    tickvals = plt.yticks()[0]
    ylabs = ["{}".format(round(v*yfact)) for v in tickvals]
    plt.gca().yaxis.set_ticklabels(ylabs)


def gen_plots(data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'linear-eval')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']

    #for axes for subplots
    i = 0
    j = 0

    #creates suplots
    #didn't use setup_plot since the aspect ratios weren't ideal for the figures
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(4, 4, figsize=(25, 16), sharey=False, constrained_layout=True)
    for pretrain_data in linear_data.pretrain_data.unique():
        for variant in ["linear-eval-lr"]: # linear_data['variant'].unique():
            with sns.axes_style("ticks"):

                #checks to make sure axes indices are in range
                if j > 3:
                    i += 1
                    j %= 4

                variant_data=linear_data[(linear_data.variant == variant) & (linear_data.pretrain_data == pretrain_data)]
                # hack to "clean up" other analyses 
                variant_data = variant_data[(linear_data.basetrain == 'MoCo Random Init') | (linear_data.basetrain == 'HPT') ]


                print(i,j)
                ax1 = axes[i,j]

                #set up x scale
                ax1.set_xscale('symlog')
                # data = data[~((data.pretrain_iters == "5000") & (data.basetrain == "MoCo Random Init"))]
                ax1.set_xticks(ticks=variant_data['pretrain_iters'].unique().astype(int))

                name_map = {
                    50: '50',
                    500: '500',
                    5000: '5K',
                    50000: '50K',
                    100000: '100K',
                    200000: '200K',
                    400000: '400K'
                }

                ori_labels = variant_data['pretrain_iters'].unique().astype(int)
                new_labels = [name_map[key] for key in ori_labels]

                ax1.set_xticklabels(labels=new_labels, minor=False, rotation=40)
                print(variant_data['pretrain_iters'].unique())

                #hard coded bn_data result

                #plot data 
                ax1.axhline(y=options['moco_transfers'][pretrain_data], c='black', linestyle='dashed', linewidth=3, label="MoCo Direct Transfer")

                sns.lineplot(ax=ax1, x='pretrain_iters', y='result', hue='basetrain', data=variant_data, marker='o', mew=0, ms=9, linewidth=4, err_style='bars')

                ax1.plot(5000, options['data_bn'][pretrain_data], "x", ms=10,  mew=3, c='red', label = "HPT-BN")



                #set title of each subplot
                ax1.set_title(str(variant_data['dataset'].iloc[0]).replace("_", " ").title())
                # handles, _ = axes[0,0].get_legend_handles_labels()

                #remove the legend generated for each subplot 
                ax1.get_legend().remove()

                #incremetns j so next plot is for suplot to the right
                j+=1


    #get handles from first subplot (handles from all of them are the same)
    handles, _ = axes[1,0].get_legend_handles_labels()

    #remove the axis lables generated from sns for all subplots
    plt.setp(axes, xlabel='')
    plt.setp(axes, ylabel='')

    #create common x-axis and y-axis labels
    plt.setp(axes[:-1, 0], ylabel='Accuracy')
    plt.setp(axes[-1, 0], ylabel='AUROC')
    plt.setp(axes[-1, :], xlabel='Pretrain Iters')


    #lables used for legend
    #       labels = ["HPT", "MoCo Target", "MoCo ImageNet Transfer", "HPT-BN"]

    #crete legend (TODO: make legend fit at bottom of figure properly)
    #     plt.legend(handles = handles,labels=labels)
    with sns.axes_style("ticks"):
        axes[0,0].legend()

    # plt.legend(title='Basetrain', labels=labels,
    #        handles=handles[1:],loc=9,
    #        fancybox=True, shadow=True, ncol=7, bbox_to_anchor=(0.5, 0.07))

    #title for entire figure
    # fig.suptitle("HPT Linear Evaluation Across 14 Datasets", fontsize=18)

    #save figure
    outplot = os.path.join(lin_dir, '{}_pretrain-linear_eval.pdf'.format("all_datasets", variant))

    print(outplot)
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
    return plt.subplots(2, 7, figsize=set_size(width))


def reduce(data):
    avg = 0
    for i, row in data.iterrows():
        avg += row.result
            
    return avg / len(data.index)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # setup plots
    # sns.set_style('white')
    sns.set_style("ticks")
    # sns.set()
    sns.set(font_scale=1.7) 

    #list of all datasets wanted
    datasets = [
                'resisc', 'ucmerced', 'viper', 'bdd', 
                'domain_net_painting', 'domain_net_clipart', 'domain_net_infograph', 'domain_net_sketch', 
                'domain_net_quickdraw', 'domain_net_real', 'flowers', 'chest_xray_kids', 
                'chexpert', 'xview', 'coco_2014', 'pascal',  
                ]

    #

    #get files for all datasets wanted
    result_files = []
    for dataset in datasets:
        result_files.append(os.path.join(args.results_dir, dataset + "_results.json"))

    print(result_files)

    #creates a datasets list to concatenate all pd to form one large pandas df
    datasets_pd = []

    #createsd dictionary for moco transfer result and bn result for each dataset
    moco_transfers = {}
    data_bn_points = {}

    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)
        #get data into pandas dataframe
        data = pd.DataFrame(raw_data.values())

        #takes only linear evals
        data = data[data.result_type =='linear-eval']

        #ignores all imagenet basetrain models
        data = data[data.basetrain != "imagenet_r50_supervised"]

        #renames the basetrain
        data.basetrain = data.basetrain.replace("moco_v2_800ep", "HPT")
        data.basetrain = data.basetrain.replace("no", "MoCo Random Init")
        data.basetrain = data.basetrain.replace("none", "MoCo Random Init")


        #ignores all imagenet basetrain models
        # data = data[~((data.pretrain_iters == "5000") & (data.basetrain == "MoCo Random Init"))]

        #gets all bn data 
        data_bn = data[data.pretrain_iters.str.contains("bn")]

        #all non-bn data
        data = data[~data.pretrain_iters.str.contains("bn")]

        #converts the pretrain iters to int
        data.pretrain_iters = pd.to_numeric(data.pretrain_iters, errors='coerce')


        #converts all results to range from 0 - 100
        if data.result.max() < 1:
            data.result *= 100
        if data_bn.result.max() < 1:
            data_bn.result *= 100


        #get moco transfer values for horixontal line on graph
        dataset_name = data.dataset.iloc[0]
        moco_transfer_val = reduce(data[(data.pretrain_iters == 0) & (data.basetrain == 'HPT')] )
        moco_transfers[dataset_name] = moco_transfer_val

        #get rid of all pretrain 0 iter runs in data
        data = data[data.pretrain_iters != 0]

        #get best data_bn value
        top_bn_val = data_bn.result.max()
        data_bn_points[dataset_name] = top_bn_val

        #adds df to a list
        datasets_pd.append(data)

    #concatenates all df into a final df
    result = pd.concat(datasets_pd, ignore_index=True)  
    print(result)
    print(moco_transfers)

    #passes in the final df, and dictionaries for moco transger and data_bn and output dir
    gen_plots(result, {
        'out_dir': args.out_dir,
        'moco_transfers' : moco_transfers,
        'data_bn': data_bn_points
    })

if __name__ == "__main__":
    main(parse_args())