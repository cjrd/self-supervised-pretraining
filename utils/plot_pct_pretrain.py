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

def logit_y_axis(yfact=100):
    # logit y axes with readable labels
    plt.yscale('logit')
    tickvals = plt.yticks()[0]
    ylabs = ["{}".format(round(v*yfact)) for v in tickvals]
    plt.gca().yaxis.set_ticklabels(ylabs)


def gen_plots(data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'pretrain_robustness')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']
    for pretrain_data in linear_data.pretrain_data.unique():
        for variant in ["linear-eval-lr"]: # linear_data['variant'].unique():
            variant_data=linear_data[(linear_data.variant == variant) & (linear_data.pretrain_data == pretrain_data)]
            # hack to "clean up" other analyses 
            variant_data = variant_data[(linear_data.basetrain == 'MoCo Random Init') | (linear_data.basetrain == 'HPT') | (linear_data.basetrain ==  'supervised imagenet init') | (linear_data.basetrain == 'HPT-BN')]
            # fig, ax = setup_plot(500)
            # print(variant_data)
            fig, ax = setup_plot(600)

            # variant_data.groupby("basetrain").plot(x="pretrain_iters", y="result", marker="o", ax=ax)
            sns.lineplot(x='pct_train', y='result', hue='basetrain', data=variant_data, marker='o', mew=0, ms=5)
            # fig.set(xlabel='Percent of Pretrain data', ylabel='Change of Accuracy from Baseline', title=options['data_name'].replace("_", " ").title())
            # plt.close(1) 
            plt.axhline(y=options['asymptote'], c='red', linestyle='dashed', label="MoCo Direct Transfer")

            plt.xlabel("Percentage of Pretrain Data")
            plt.ylabel("Accuracy")
            plt.title(options['data_name'].replace("_", " ").title())



            plt.xscale('linear')
            plt.xticks(ticks=variant_data['pct_train'], labels=variant_data['pct_train'], rotation='horizontal')

            labels = ["HPT", "HPT-BN", "MoCo Random Init", "MoCo Direct Transfer"]
            handles, _ = ax.get_legend_handles_labels()
            plt.legend(handles = handles,labels=labels,title="Basetrain")
            plt.show()
            outplot = os.path.join(lin_dir, '{}_data_robustness-{}_linear_eval.pdf'.format(pretrain_data, options['data_name']))
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

def reduce(data):
    best = 0
    for i, row in data.iterrows():
        if best < row['result']:
            best = row['result']
            
    return data[data.result==best]


def setup_plot(width=300):
    return plt.subplots(1, 1, figsize=set_size(width))


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

    #gets all files that start with "*pct" for the corresponding dataset
    result_files = glob.glob(os.path.join(args.results_dir, dataset_type + "*pct*.json"), recursive=True)
    result_files.append(os.path.join(args.results_dir, dataset_type + "_results.json")) #gets file with 100% pretrain data results
    if (dataset_type != 'resisc'):
        result_files.append(os.path.join(args.results_dir, dataset_type + "_bn_results.json"))
    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)

        print(resfile)

        types = [] #used for concatenating the resultzs from each basetrained model
        data = pd.DataFrame(raw_data.values())


        #finds the relevant values for moco bt, and no bt
        #mainly done to work around the baseline json file (has extra info that we don't want)
        if (dataset_type + "_results.json") in resfile:
            linear_data = data[data.result_type=='linear-eval']
            linear_data = linear_data[linear_data.variant=="linear-eval-lr"]

            data_moco = linear_data[data.basetrain=="moco_v2_800ep"]
            # data_moco = data_moco[data_moco.pretrain_iters=="5000"]
            data_moco = data_moco[data_moco.pretrain_data==dataset_type]
            data_moco = reduce(data_moco)


            data_nobt = linear_data[data.basetrain=="no"]
            # data_nobt= data_nobt[data_nobt.pretrain_iters=="100000"]
            data_nobt= data_nobt[data_nobt.pretrain_data==dataset_type]
            data_nobt = reduce(data_nobt)


            #appended to list 
            types.append(data_moco)
            types.append(data_nobt)

            #all concat to new dataframe 
            data = pd.concat(types, ignore_index=True)
            print(data)

        if "bn" in resfile:
            data_bn = data[data.pretrain_iters!='0']
            data.basetrain = data.basetrain.replace("moco_v2_800ep", "HPT-BN")

        #convert pretrain iters values to int

        data = data.astype({
            "pretrain_iters": int
        })

        #multiply all results < 1 by 100
        if not (data.result > 1).all():
            data.result *=100


        #change corresponding names of each basetrain 
        data.basetrain = data.basetrain.replace("moco_v2_800ep", "HPT")
        data.basetrain = data.basetrain.replace("no", "MoCo Random Init")
        data.basetrain = data.basetrain.replace("none", "MoCo Random Init")    
        frames.append(data)

    #combines all the dataframes from each file into 1 large dataframe
    result = pd.concat(frames, ignore_index=True)

    #add a column for pct pretrain data and iterate through each row to add appropriate value
    pct = [0]*len(result)
    asymptote = 0
    count = 0
    result["pct_train"] = pct
    
    for i, row in result.iterrows():
        val = 0
        if row["pretrain_iters"] == 0:
            asymptote +=row["result"]
            count+=1
            continue
        if "1_pct" in row["dataset"]:
            val = 1
        elif "10_pct" in row["dataset"] :
            val = 10
        elif "25_pct" in row["dataset"]:
            val = 25
        else:
            val = 100
        result.at[i,"pct_train"] = val

    result = result[result['pretrain_iters'] > 0] 
    result.pretrain_data = dataset_type

    #changes entries to best performace 
    pct1 = result[(result['pct_train'] == 1) & (result['basetrain'] == "HPT")]
    pct1.result = pct1['result'].max()
    pct10 = result[(result['pct_train'] == 10) & (result['basetrain'] == "HPT")]
    pct10.result = pct10['result'].max()
    pct25 = result[(result['pct_train'] == 25) & (result['basetrain'] == "HPT")]
    pct25.result = pct25['result'].max()

    moco = pd.concat([pct1,pct10,pct25], ignore_index=True)

    pct1_bn = result[(result['pct_train'] == 1) & (result['basetrain'] == "HPT-BN")]
    pct1_bn.result = pct1_bn['result'].max()
    pct10_bn = result[(result['pct_train'] == 10) & (result['basetrain'] == "HPT-BN")]
    pct10_bn.result = pct10_bn['result'].max()
    pct25_bn = result[(result['pct_train'] == 25) & (result['basetrain'] == "HPT-BN")]
    pct25_bn.result = pct25_bn['result'].max()
    pct100_bn = result[(result['pct_train'] == 100) & (result['basetrain'] == "HPT-BN")]
    pct100_bn.result = pct100_bn['result'].max()

    bn = pd.concat([pct1_bn,pct10_bn,pct25_bn,pct100_bn], ignore_index=True)




    #keeps all the random inits in result pd
    result = result[((result['basetrain'] != "HPT") & (result['basetrain'] != "HPT-BN")) | (result['pct_train'] == 100)]


    #hack for resisc bn 100% pretrain
    if dataset_type == 'resisc':
        pct100_bn = result[(result['pct_train'] == 100) & (result['basetrain'] == "HPT")]
        pct100_bn.basetrain = 'HPT-BN'
        pct100_bn.result = 92.50794
        bn = pd.concat([bn,pct100_bn],ignore_index=True)

    result = pd.concat([moco,bn,result], ignore_index=True)




    #prints new concatenated dataframe 
    print(result)
    print(result.result)
    print(result.basetrain)


    dataname = dataset_type+"_data_robustness"
    asymptote /= count
    print("moco transfer:", asymptote)

    gen_plots(result, {
        'out_dir': args.out_dir,
        'data_name': dataname,
        'asymptote': asymptote
    })

if __name__ == "__main__":
    main(parse_args())