import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
import numpy as np

lines = [line.strip() for line in open(".tmp_exp3_result.txt")]

records = []
dct = {}
for line in lines:
    if line.startswith("===="):
        if len(records) > 0:
            dct[ds_name] = records
#         print(line)
        ds_name = line.split(" ")[1] + "-" + line.split(" ")[2]
        records = []
    else:
#         print(line)
        a, b = ' '.join(line.split()).split(" ") 
        records.append([b, a])

dct[ds_name] = records
mpl.style.use('seaborn')
for ss in range(2):
    for idx, key in enumerate(dct.keys()):
        fig = plt.figure(figsize=(9,6))
        vals = dct[key]
        if ss == 0:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][:9]
            names = [item[0] for item in vals][:9]
        else:
            ys = [float(item[1]) if item[1] != 'null' else 0 for item in vals][9:]
            names = [item[0] for item in vals][9:]
            
        print(names)
        print(ys)
        axes = []
        ax1 = fig.add_subplot(111)
        for i in range(1, len(names)):
            axes.append( ax1.bar(i, (ys[i] - ys[0]) if ys[i] - ys[0]>-30 else 0, color=f'C{i}', edgecolor='black') )
        
        names_new = ['_'.join(name.split('_')[2:]) for name in names]
        names_new = [name.replace('source', 'src').replace('target', 'tgt').replace('imagenet', 'sup').replace('moco', 'ssl') for name in names_new]
        plt.xticks(np.arange(1, 9), names_new[1:], rotation=-40)
        
#         print()
        split = names[0].split("_")[0]
        ax1.set_title(key + f" ({split})", fontsize = 14, fontweight ='bold')
        
        plt.ylabel("Accuracy gain")
#         plt.show()
        
        os.makedirs("./plot-results/exp-3/", exist_ok=True)
        
        plt.savefig(f"./plot-results/exp-3/exp-3-{key}_{split}.pdf", bbox_inches='tight')
