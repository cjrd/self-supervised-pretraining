
import glob
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

def logit_y_axis(yfact=100):
    # logit y axes with readable labels
    plt.yscale('logit')
    tickvals = plt.yticks()[0]
    ylabs = ["{}".format(round(v*yfact)) for v in tickvals]
    plt.gca().yaxis.set_ticklabels(ylabs)


def gen_plots(data, bn_data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'linear-eval')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']
    for pretrain_data in linear_data.pretrain_data.unique():
        for variant in ["linear-eval-lr"]: # linear_data['variant'].unique():
            variant_data=linear_data[(linear_data.variant == variant) & 
                (linear_data.pretrain_data == pretrain_data) & 
                (linear_data.pretrain_iters > 0) &
                (linear_data.basetrain != "supervised imagenet init")
                ]

            moco_zero_val = linear_data[(linear_data.pretrain_iters ==0) & (linear_data.basetrain == "moco imagenet init")].result.mean()
            top_bn_val = bn_data.result.max()
            if top_bn_val < 1:
                top_bn_val *= 100

            # hack to "clean up" other analyses 
            variant_data = variant_data[(linear_data.basetrain == 'random init') | (linear_data.basetrain == 'moco imagenet init') | (linear_data.basetrain ==  'supervised imagenet init') ]
            fig, ax = setup_plot(500)

            # variant_data.groupby("basetrain").plot(x="pretrain_iters", y="result", marker="o", ax=ax)
            sns.lineplot(x='pretrain_iters', y='result', hue='basetrain', data=variant_data, marker='o', mew=0, ms=5)
            ax.axhline(moco_zero_val, ls='--', label="moco transfer")

            # TODO(cjrd) hardcoding
            ax.plot(5000, top_bn_val, "xr", label="HPT-BN")

            labels = ["HPT", "MoCo Random Init", "MoCo Direct Transfer", "HPT-BN"]
            handles, _ = ax.get_legend_handles_labels()

            # Slice list to remove first handle
            plt.legend(handles = handles, labels = labels)
            
            plt.xlabel("Pretrain Steps")
            plt.ylabel("Accuracy")
            plt.title(options['data_name'].replace("_", " ").title())

            plt.xscale('log')
            plt.xticks(ticks=variant_data['pretrain_iters'], labels=variant_data['pretrain_iters'], rotation='vertical')

            # logit_y_axis()
            outplot = os.path.join(lin_dir, '{}_pretrain-{}_linear_eval-{}_variant.pdf'.format(pretrain_data, options['data_name'], variant))
            print(outplot)
            fig.savefig(outplot, format='pdf', bbox_inches='tight')


    # finetune plots
    finetune_dir = os.path.join(options['out_dir'], 'finetune-eval')
    os.makedirs(finetune_dir, exist_ok=True)
    finetune_data = data[data.result_type=='finetune']

    for subset in finetune_data.subset.unique():
        for pretrain_data in linear_data.pretrain_data.unique():
            subset_data = finetune_data[(finetune_data.subset == subset) & (finetune_data.pretrain_data == pretrain_data)]
            # take max val across each sample, iter, and basetrain
            # (isn't pandas cool?)
            subset_data = subset_data.groupby(['sample', 'pretrain_iters', 'basetrain']).apply(lambda x: x.loc[x.result.idxmax()])
            if subset_data.empty:
                continue
            
            fig, ax = setup_plot(500)

            sns.lineplot(x='pretrain_iters', y='result', hue='basetrain', data=subset_data, marker='o', mew=0, ms=5)

            plt.xlabel("Pretrain Steps")
            plt.ylabel("Accuracy")
            plt.xscale('symlog')
            plt.xticks(ticks=subset_data['pretrain_iters'], labels=subset_data['pretrain_iters'], rotation='vertical')
            logit_y_axis()
            
            plt.title(options['data_name'].replace("_", " ").title())

            outplot = os.path.join(finetune_dir, '{}_pretrain-{}_fintune-{}_subset.pdf'.format(pretrain_data, options['data_name'], subset))
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
    return plt.subplots(1, 1, figsize=set_size(width))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # setup plots
    sns.set_style('darkgrid')
    sns.set()

    result_files = glob.glob(os.path.join(args.results_dir, "*.json"), recursive=True)
    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)
        data = pd.DataFrame(raw_data.values())
        dataname = data.dataset[0]
        if args.dataset != "all" and dataname != args.dataset:
             continue

        bn_data = data[data.pretrain_iters.str.contains("bn")]
        data = data[~data.pretrain_iters.str.contains("bn")]        
        data.pretrain_iters = pd.to_numeric(data.pretrain_iters, errors='coerce')
        # convert batchnorm iters to a new type of data

        try:
            data = data.astype({
                "pretrain_iters": int
            })
        # pokemon exceptions!
        except Exception as exp:
            print(f"WARNING: Unable to parse pretrain_iters as int for {dataname}")
        if data.result.max() < 1:
            data.result *= 100
        data.basetrain = data.basetrain.replace("imagenet_r50_supervised", "supervised imagenet init")
        data.basetrain = data.basetrain.replace("moco_v2_800ep", "moco imagenet init")
        data.basetrain = data.basetrain.replace("no", "random init")
        data.basetrain = data.basetrain.replace("none", "random init")        
        gen_plots(data, bn_data, {
            'out_dir': args.out_dir,
            'data_name': dataname
        })

if __name__ == "__main__":
    main(parse_args())