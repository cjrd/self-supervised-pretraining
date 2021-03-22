import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
import numpy as np

lines = [line.strip() for line in open(".tmp_result.txt")]

records = []
dct = {}
for line in lines:
    if line.startswith("#####"):
        if len(records) > 0:
            dct[ds_name] = records
        ds_name = line.split(" ")[1]
        records = []
    else:
        a, b = ' '.join(line.split()).split(" ") 
        records.append([b, a])

dct[ds_name] = records

mpl.style.use('seaborn')

fig = plt.figure(figsize=(45,10))
N = len(dct.keys())
num_cols = 6
for ss in range(2):
    for idx, key in enumerate(dct.keys()):
        fig = plt.figure(figsize=(9,4))
        vals = dct[key]
        if ss == 0:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][:6]
            names = [item[0] for item in vals][:6]
        else:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][6:]
            names = [item[0] for item in vals][6:]          

        patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "/" , "\\" , "|" , "-" , "+" , "x"]
        ax1 = fig.add_subplot(111)
        axes = []
    #     print("ys: ", ys)
        logits = [math.log((val/100.) / (1 - val/100.)) if val!=0 else 66 for val in ys]
    #     print("logits: ", logits)
        for i in range(len(names)):
            if i == 0:
                continue
            acc_inc = ys[i] - ys[0]
            acc_inc = logits[i] - logits[0] if logits[i] != 66 else 0
    #         logit = math.log(acc_inc / (1 - acc_inc), 10)
            axes.append( ax1.bar(i, acc_inc, color=f'C{i}', edgecolor='black', hatch=patterns[i]) )

        ax1.set_xticklabels(names, rotation=-40)

        split = names[0].split("_")[0]
        ax1.set_title(key + f" ({split})", fontsize = 14, fontweight ='bold')
        plt.ylabel("Logit accuracy gain")
#         plt.show()

        os.makedirs("./plot-results/exp-2/", exist_ok=True)
        
        plt.savefig(f"./plot-results/exp-2/exp-2-{key}_{split}.pdf", bbox_inches='tight')






# Plot the new combined ones
lines = [line.strip() for line in open(".tmp_result.txt")]

records = []
dct = {}
for line in lines:
    if line.startswith("#####"):
        if len(records) > 0:
            dct[ds_name] = records
        ds_name = line.split(" ")[1]
        records = []
    else:
        a, b = ' '.join(line.split()).split(" ") 
        records.append([b, a])

dct[ds_name] = records

mpl.style.use('seaborn')

fig = plt.figure(figsize=(45,10))
N = len(dct.keys())
num_cols = 6
for ss in range(2):
    for idx, key in enumerate(dct.keys()):
        fig = plt.figure(figsize=(9,6))
        vals = dct[key]
        if ss == 0:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][:6]
            names = [item[0] for item in vals][:6]
        else:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][6:]
            names = [item[0] for item in vals][6:]          

#         patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "/" , "\\" , "|" , "-" , "+" , "x"]
        # (241/255., 156/255., 56/255.) (235/255., 213/255., 184/255.) (17/255., 56/255., 95/255.) (159/255., 235/255., 234/255.)
        bot_colors = [(241/255., 156/255., 86/255.), (117/255., 156/255., 195/255.), (32/255., 126/255., 26/255.)]
        ax1 = fig.add_subplot(111)
        axes = []
#         logits = [math.log((val/100.) / (1 - val/100.)) if val!=0 else 66 for val in ys]
        logits = [val if val!=0 else 666 for val in ys]
        for i in range(len(names)):
            if i == 0:
                continue
#             acc_inc = ys[i] - ys[0]
            acc_inc = logits[i] - logits[0] if logits[i] != 666 else 0
    #         logit = math.log(acc_inc / (1 - acc_inc), 10)
            
            # , hatch=patterns[i]
            if i != 3:
                axes.append( ax1.bar(i, acc_inc, color=bot_colors[i-1], edgecolor='black') )
                acc_inc_1 = (logits[i] - logits[0]) if logits[i] != 666 else 0
                acc_inc_2 = (logits[i+3] - logits[0]) if logits[i+3] != 666 else 0
                
                # , hatch=patterns[i+3]
                if acc_inc_2 - acc_inc_1 > 0:
                    ax1.bar(i, acc_inc_2 - acc_inc_1, color=bot_colors[-1], edgecolor='black', bottom=acc_inc_1)
                else:
                    ax1.bar(i, acc_inc_1 - acc_inc_2, color='red', edgecolor='black', bottom=acc_inc_2)
            if i == 3:
                plt.axhline(acc_inc, color='black', dashes=(5, 2, 0, 2))
                break

#         ax1.set_xticklabels([""]*2+names[1:-2], rotation=-40)
        plt.xticks(np.arange(1,3), names[1:-3], rotation=-40)

        split = names[0].split("_")[0]
        ax1.set_title(key + f" ({split})", fontsize = 14, fontweight ='bold')
        
        plt.ylabel("Accuracy gain")
        # plt.show()
#         break
#     break

        os.makedirs("./plot-results/exp-2/", exist_ok=True)
        
        plt.savefig(f"./plot-results/exp-2/exp-2-{key}_{split}_new.pdf", bbox_inches='tight')
