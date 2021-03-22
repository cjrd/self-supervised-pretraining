import argparse
import json
import os
import pwd
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Parse agg results files into a friendly formatted results json for each dataset')
    parser.add_argument(
        '--agg-file',
        required=True)
    parser.add_argument(
        '--outdir',
        required=True)

    args = parser.parse_args()
    return args


def str_to_result(inp_str, author=None):
    result = {}
    inp_fields = inp_str.split("/")

    # WARNING: the filenames are brittle, so this is, err, brittle
    result["dataset"] = inp_fields.pop(0).split("-")[-1]
    result["result_type"] = inp_fields.pop(0)

    if result["result_type"] == "finetune":
        result["subset"] = inp_fields.pop(0)
    else:
        result["subset"] = "all"
        
    # "intelligently" parse the basetrain field using hyphens
    inp_fields[0] = inp_fields[0].replace('-21352794', '_21352794')
    inp_fields[0] = inp_fields[0].replace('final_backbone_basetrain-', '')
    
    hierarchy = inp_fields.pop(0).split("-")
    if len(hierarchy) == 1:
        result["basetrain"] = hierarchy[0].replace("_basetrain", "")
        result["pretrain_data"] = result["dataset"]
    else:
        steps = [v.replace("_basetrain", "").replace("no", "none") for v in hierarchy]
        # TODO(cjrd) generalize this
        if len(steps) > 2:
            steps = ["-".join(steps[:-1]), steps[-1]] 
        result["basetrain"] = steps[0]
        result["pretrain_data"] = steps[1]

    # "int3lligently" parse the model weight string
    details = inp_fields.pop(0).split("-")
    result["pretrain_iters"] = details[0]
    # check if the last bit is s0 or s1 etc
    if details[-1][0] == "s" and len(details[-1]) == 2:
        result["sample"] = int(details.pop()[-1])

    else:
        result["sample"] = 0

    result["variant"] = "-".join(details[2:])

    # remaining str: date result file
    remaining_str = inp_fields.pop(0)
    result["date"] = remaining_str.split(".")[0]
    try:
        result["result"] = float(remaining_str.split(":::")[-1].split()[0].strip())
    except:
        # some don't have results
        print("WARNING: unable to parse result from: {}".format(inp_str))
    result["file"] = inp_str.split(":::")[0].strip()

    # result["author"] = author

    return result

if __name__ == "__main__":
    args = parse_args()
    author=pwd.getpwuid(os.getuid()).pw_name
    with open(args.agg_file, 'r') as reslist:
        results = [str_to_result(l, author) for l in reslist]

    os.makedirs(args.outdir, exist_ok=True)
    for dset in set([r['dataset'] for r in results]):
        print("saving results for dataset: {}".format(dset))
        dset_res = {r["file"]: r for r in results if r['dataset'] == dset}
        with open(os.path.join(args.outdir,'{}_results.json'.format(dset)), 'w') as ofile:
            json.dump(dset_res, ofile)
    # now separate base on dataset and save to different json files in output dir


