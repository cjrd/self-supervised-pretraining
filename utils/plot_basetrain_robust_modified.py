
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
        default=os.path.realpath(os.path.join(dir_path, "results_modified")))
    parser.add_argument(
        '--out-dir',
        default=os.path.realpath(os.path.join(dir_path, "plot-results")))
    parser.add_argument(
        '--dataset',
        default="all")

    args = parser.parse_args()
    return args


#only deals with linear plots 
def gen_plots(data, options):

    # linear plots
    lin_dir = os.path.join(options['out_dir'], 'basetrain_robustness')
    os.makedirs(lin_dir, exist_ok=True)
    linear_data = data[data.result_type=='linear-eval']
    linear_data = linear_data[linear_data.variant=="linear-eval-lr"]

    #since catplot is a figure level function, it produces a new, separate plot which doesn't follow style of past graphs
    fig= sns.catplot(x='basetrain', y='result', hue='basetrain_robust', data=linear_data, kind="point",s=10,
        linestyle="-", legend_out=False, order=["MoCo 20-epochs","MoCo 200-epochs","MoCo 800-epochs"], aspect=11.7/8.27)
    plt.axhline(y=options['asymptote'], c='red', linestyle='dashed', label="Best MoCo Random Init")
    
    #sets axis labels and title of graph
    y_axis = "Accuracy"
    if options['dataset_type'] == 'chexpert':
        y_axis = "AUROC"

    fig.set(xlabel='Pretrained Model', ylabel=y_axis, title=options['data_name'].replace("_", " ").title())
    # fig.set_yticklabels()
    #sets legend
    fig.ax.legend(fontsize="large")
    
    
    
    for ax in fig.axes.flat:
        if options['dataset_type'] != 'chexpert':
            ax.set_yticks(ax.get_yticks()[::2])
        ax.set_title(options['data_name'][:-21].replace("_", " ").title(), fontsize=25)
        ax.set_xlabel("Augmentation Sets", fontsize=20)
        ax.set_ylabel("Accuracy Change", fontsize=20)
        xticks = ["20-epochs","200-epochs","800-epochs"]
        ax.set_xticklabels(xticks, rotation=0, fontsize=20)

        yticks = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels(yticks, rotation=0, fontsize=20)
        # ax.tick_params(axis='y', labelsize='large')
        ax.spines['bottom'].set_color('1')
        ax.spines['top'].set_color('1')
        ax.spines['right'].set_color('1')
        ax.spines['left'].set_color('1')
        ax.patch.set_facecolor('0.97')
        ax.grid(axis='y', color='grey', dashes=[10, 4])
    # handles, _ = fig.ax.get_legend_handles_labels()
    # fig.ax.legend(handles = handles[1:],labels=labels,title="Basetrain")
    # plt.show()

    #gets rid of other graph created 
    # plt.close(1) 

    #creates output file
    outplot = os.path.join(lin_dir, '{}.pdf'.format(options['data_name']))

    #saves Figure 1, but Figure 1 is the empty graph created (NOT SNS Graph)
    fig.savefig(outplot, format='pdf', bbox_inches='tight')


def reduce(data):
    best = 0
    for i, row in data.iterrows():
        if best < row['result']:
            best = row['result']
    return best

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

    nobt_baseline = 0

    #gets all the basetrain results from the specified dataset
    result_files = glob.glob(os.path.join(args.results_dir, dataset_type + "*basetrain*.json"), recursive=True)
    result_files2 = glob.glob(os.path.join(args.results_dir, dataset_type + "*bt_robust*.json"), recursive=True)
    result_files += result_files2
    result_files.append(os.path.join(args.results_dir, dataset_type + "_results.json")) #gets file with basetrian results

    for resfile in result_files:
        with open(resfile, 'r') as infile:
            raw_data = json.load(infile)

        print(resfile)

        types = [] #used for concatenating the resultzs from each basetrained model
        data = pd.DataFrame(raw_data.values())


        #finds the relevant values for moco bt no bt
        #mainly done to work around the baseline json file (has extra info that we don't want)
        if (dataset_type + "_results.json") in resfile:
            linear_data = data[data.result_type=='linear-eval']
            linear_data = linear_data[linear_data.variant=="linear-eval-lr"]

            data_moco = linear_data[data.basetrain=="moco_v2_800ep"]
            data_moco = data_moco[data_moco.pretrain_data==dataset_type]
            data_moco_x = data_moco[(data_moco['pretrain_iters']=="5000")]
            data_moco_y = data_moco[(data_moco['pretrain_iters']=="0")]
            data_moco_x = data_moco_x.sort_values(by=['result'],ascending=False).head(1)
            data_moco_y = data_moco_y.sort_values(by=['result'],ascending=False).head(1)
            data_moco = pd.concat([data_moco_x,data_moco_y],ignore_index=True)

            #just for asympotote
            data_nobt = linear_data[data.basetrain=="no"]
            data_nobt= data_nobt[data_nobt.pretrain_data==dataset_type]
            nobt_baseline = reduce(data_nobt)


            #appended to list 
            types.append(data_moco)

            #all concat to new dataframe 
            data = pd.concat(types, ignore_index=True)

        #get baseline and update results

        data = data.astype({
            "pretrain_iters": int
        })
        if not (data.result > 1).all():
            data.result *=100

        data.basetrain = data.basetrain.replace("moco_v2_800ep", "MoCo 800-epochs")
        data.basetrain = data.basetrain.replace("moco_v2_200ep_pretrain", "MoCo 200-epochs")
        data.basetrain = data.basetrain.replace("moco_v2_20ep_pretrain", "MoCo 20-epochs")
        data.basetrain = data.basetrain.replace("no", "random init")
        data.basetrain = data.basetrain.replace("none", "random init")
        frames.append(data)

    #combines all the dataframes from each file into 1 large dataframe
    result = pd.concat(frames, ignore_index=True)
    dataname = dataset_type + "_basetrain_robustness"
    asymptote = nobt_baseline
    if asymptote < 1:
        asymptote *=100
    print(result)
    print(result.result)
    print(asymptote)

    basetrain_robust = ["a"]*len(result)
    result["basetrain_robust"] = basetrain_robust

    print(result.pretrain_iters)
    for i, row in result.iterrows():
        name = ""
        if row['pretrain_iters'] == 0:
            name = "MoCo Direct Transfer"
        else:
            name = "HPT"

        result.at[i,"basetrain_robust"] = name
    print(result)

    #only ran extra experiments for moco_20
    #take only the best result (50k iterations)
    result_20_m = result[(result['basetrain'] == "MoCo 20-epochs") & (result['basetrain_robust']== "HPT")].sort_values(by=['result'], ascending=False).head(1)
    
    result_20_o = result[(result['basetrain'] == "MoCo 20-epochs") & (result['basetrain_robust']== "MoCo Direct Transfer")].sort_values(by=['result'], ascending=False).head(1)

    result_other = result[((result['basetrain'] != "MoCo 20-epochs") | (result['basetrain_robust']== "MoCo Direct Transfer")) & ((result['basetrain'] != "MoCo 20-epochs") | (result['basetrain_robust']!= "MoCo Direct Transfer"))]
    result = pd.concat([result_20_m,result_20_o,result_other], ignore_index=True)


    #changes all the dataset names to dataset_augmentation
    
    result.dataset = dataset_type

    #prints new concatenated dataframe 
    print(result)
    print(result.result)
    
    gen_plots(result, {
        'out_dir': args.out_dir,
        'data_name': dataname,
        'asymptote': asymptote,
        'dataset_type': dataset_type
    })

if __name__ == "__main__":
    main(parse_args())