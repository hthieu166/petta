import pandas as pd
import argparse
import yaml
from pathlib import Path
import numpy as np
import os
import os.path as osp

ROUND_DIGIT=1

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment_lst", type=str, default="exp_tables.yaml")
    parser.add_argument("--out_dir", type=str, default="tex_tables")
    
    return parser.parse_args()
args = parse_args()

# Get list of experiments
lst_file = yaml.safe_load(Path(args.experiment_lst).read_text())

def read_eval_table(csv_file):
    res = {}
    df = pd.read_csv(csv_file)
    if "domainnet" in csv_file:
        for i in range(len(df)):
            if df["Corruption"].iloc[i] in ["clipart", "painting", "sketch"]:
                df["Corruption"].iloc[i] = f'{df["Corruption"].iloc[i]}_rep0'
    if "imagenetc" in csv_file:
        for i in range(len(df)):
            if "_rep" not in df["Corruption"].iloc[i]:
                df["Corruption"].iloc[i] = f'{df["Corruption"].iloc[i]}_rep0'
    
    for i in range(len(df)):
        nm = df["Corruption"].iloc[i]
        if "_rep" in nm: 
            acc = df["Error Rate"].iloc[i]
        
            rep = int(nm.split("_rep")[-1]) + 1
            if rep not in res:
                res[rep] = []
            res[rep].append(acc)
    for k in res:
        res[k] = np.array(res[k]).mean()
    return res

def export_eval_table(cfgs, annotate="latex"):
    for dts in cfgs:
        print(dts)
        eval_dat = {}
        for inf in cfgs[dts]:
            print(inf)
            name = inf["name"]
            res = inf["fold"]
            eval_dat[name] = read_eval_table(f'{res}.csv')

        df = pd.DataFrame.from_dict(eval_dat, orient="index")
        df["\textbf{Avg}" if annotate == "latex" else "**Avg**"] = df.mean(axis=1)
        df = df.round(ROUND_DIGIT)
        
        if not osp.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        # To CSV
        df.to_csv( f"tex_tables/{dts}.csv", sep=",")

if __name__ == "__main__":
    cfgs = yaml.safe_load(Path(args.experiment_lst).read_text())
    # Export to latext tables
    export_eval_table(cfgs, annotate="latex")