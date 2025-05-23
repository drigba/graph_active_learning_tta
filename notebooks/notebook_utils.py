import os
import pandas as pd
from graph_al.test_time_adaptation.config import AdaptationConfig
import plotly.graph_objects as go
import torch
import numpy as np

def load_results(dataset, model, strategies_names,save = False, cached = False, cache_path = None, verbose = False):
    
    if cached and cache_path is not None:
        print("Loading cached metrics")
        if not os.path.exists(cache_path):
            raise ValueError(f"Cache path {cache_path} does not exist.")
        metrics_path = os.path.join(cache_path,f"{dataset}_{model}_metrics_dict.pt")
        metrics_dict = torch.load(metrics_path, weights_only=False)
        return metrics_dict
    
    prefix = f"../output2/runs/{dataset}/{model}/"

    strategies_paths = [os.path.join(prefix, strategy) for strategy in strategies_names if os.path.exists(os.path.join(prefix, strategy))]
    strategies_names_filtered = [strategy for strategy in strategies_names if os.path.exists(os.path.join(prefix, strategy))]
    metrics_dict = {}
    print(f"Loading metrics {dataset} {model}")
    for ix,strategies_path in enumerate(strategies_paths):
        print(f"\t{strategies_names_filtered[ix]} metrics")
        strategies = os.listdir(strategies_path)
        for strategy in strategies:
            path = os.path.join(strategies_path, strategy)
            for run in os.listdir(path):
                if os.path.exists(os.path.join(path, run, "acquisition_curve_metrics.pt")):
                    metrics_path = os.path.join(path, run, "acquisition_curve_metrics.pt")
                    if verbose:
                        print(metrics_path)
                    metrics = torch.load(metrics_path, weights_only=True)
                    accuracy = np.array(metrics["accuracy/test"])*100
                    accuracy_mean, accuracy_std = np.mean(accuracy, axis=0), np.std(accuracy, axis=0)
                    metrics_dict[strategies_names_filtered[ix] + "_" + strategy] = (accuracy_mean,accuracy_std,accuracy,metrics)
    if save and cache_path is not None:
        print("Saving metrics to cache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        torch.save(metrics_dict, os.path.join(cache_path,f"{dataset}_{model}_metrics_dict.pt"))
    return metrics_dict

def create_tta_pivots(tta_df,acc_col):
    tta_df_mask = tta_df[(tta_df["m_f"] == "mask") & (tta_df["num"] == 100)]
    tta_df_mask_pivot = tta_df_mask.pivot(index="p_f", columns="p_e", values=acc_col)
    tta_df_mask_pivot.sort_index(axis=0, ascending=False, inplace=True)
    
    tta_df_noise = tta_df[(tta_df["m_f"] == "noise") & (tta_df["num"] == 100)]
    tta_df_noise_pivot = tta_df_noise.pivot(index="p_f", columns="p_e", values=acc_col)
    tta_df_noise_pivot.sort_index(axis=0, ascending=False, inplace=True)
    
    vmin = min(tta_df_mask_pivot.min().min(), tta_df_noise_pivot.min().min())
    vmax = max(tta_df_mask_pivot.max().max(), tta_df_noise_pivot.max().max())
    
    return tta_df_mask_pivot, tta_df_noise_pivot, vmin, vmax



def update_progress(df, path):
    for dataset in os.listdir(path):
        for model in os.listdir(os.path.join(path, dataset)):
            for strategy in os.listdir(os.path.join(path, dataset, model)):
                df.loc[(dataset, model, strategy), "progress"] = True
                strat_path = os.path.join(path, dataset, model, strategy)
                cnt = 0
                for seed in os.listdir(strat_path):
                    seed_dir = os.path.join(strat_path, seed)
                    for run_dir in os.listdir(seed_dir):
                        run_path = os.path.join(seed_dir, run_dir)
                        if run_dir != "hydra-outputs":
                            if strategy == "geem":
                                if os.path.exists(os.path.join(run_path, "acquisition_metrics-4-4-24.pt")):
                                    cnt = 25
                                    break
                                elif os.path.exists(os.path.join(run_path, "acquisition_curve_metrics.pt")):
                                    cnt += 1
                            else:
                                cnt = max(cnt, sum(1 for file in os.listdir(run_path) if file.startswith("acquisition_metrics")))
                df.loc[(dataset, model, strategy), "progress_percentage"] = cnt / 25
                df.loc[(dataset, model, strategy), "progress_count"] = cnt
                if cnt >= 25:
                    df.loc[(dataset, model, strategy), "done"] = True
    return df

def plot_progress(expected_datasets, df):
    for dataset in expected_datasets:
        df_reset = df.reset_index()
        df_reset = df_reset[df_reset["dataset"] == dataset]
        df_reset["strategy"] = df_reset["strategy"].apply(lambda x: x.replace("approximate_uncertainty_", ""))
        df_reset.loc[df_reset["strategy"] == "aleatoric_propagated","strategy"] = "aleatoric"
        fig = go.Figure()
        fig.add_bar(x=df_reset.transpose().loc[["model","strategy"]],y=df_reset["progress_percentage"].transpose())
        fig.update_layout(title=dataset.capitalize())
        fig.show()
        
        
def generate_prompt(dataset, model, strategy):
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=5 "
        f"model.num_inits=5 print_summary=True "
        f"model.cached=True "
        f"acquisition_strategy.adaptation_enabled=False "
        f"acquisition_strategy.tta_enabled=False > logs_new/{dataset}_{model}_{strategy}.log &"
    )
    return s


def generate_prompt_geem(dataset, model, strategy, seed):
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=1 "
        f"acquisition_strategy.adaptation_enabled=False "
        f"model.num_inits=1 print_summary=True "
        f"seed={seed} "
        f"model.cached=True "
        f"wandb.name={seed} "
        f"acquisition_strategy.tta_enabled=False > logs_new/{dataset}_{model}_{strategy}_{seed}.log &"
    )
    return s


from graph_al.acquisition.enum import *

def generate_adaptation_name(adaptation_config: AdaptationConfig):
    match adaptation_config.mode:
        case AdaptationMode.FEATURE:
            return f"feature_lr{adaptation_config.lr_feat}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        case AdaptationMode.STRUCTURE:
            return f"adj_lr{adaptation_config.lr_adj}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        case AdaptationMode.BOTH:
            return f"graph_lra{adaptation_config.lr_adj}_lrf{adaptation_config.lr_feat}_epochs{adaptation_config.epochs}_i{adaptation_config.integration}"
        
        
        
def generate_prompt_adaptation(dataset,model,strategy,adaptation_config:AdaptationConfig,scale, seed):
    wn = generate_adaptation_name(adaptation_config)
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits=5 "
        f"model.num_inits=5 print_summary=True "
        f"model.cached=False "
        f"seed={seed} "
        f"acquisition_strategy.tta_enabled=False "
        f"acquisition_strategy.adaptation_enabled=True "
        f"acquisition_strategy.scale={scale} "
        f"acquisition_strategy.adaptation.lr_feat={adaptation_config.lr_feat} "
        f"acquisition_strategy.adaptation.lr_adj={adaptation_config.lr_adj} "
        f"acquisition_strategy.adaptation.epochs={adaptation_config.epochs} "
        f"acquisition_strategy.adaptation.seed={adaptation_config.seed} "
        f"acquisition_strategy.adaptation.mode={adaptation_config.mode} "
        f"acquisition_strategy.adaptation.integration={adaptation_config.integration} "
        f"wandb.name={wn} "
        f"> logs_new/{dataset}_{model}_{strategy}_adaptation_{wn}.log &"
    )
    p = f"{dataset}/{model}/{strategy}/{wn}"
    return s,p


def generate_prompt_tta(dataset,model,strategy,strat_node, strat_edge,num, filter,probs,scale, seed, p_node = "none", p_edge = "none", n_splits = 5, n_inits = 5):
    f = "filter" if filter else "nofilter"
    pl = "probs" if probs else "logits"
    seed_text = ("_" + str(seed)) if int(seed) != 42 else "" 
    node_prob = f"acquisition_strategy.tta.p_node={p_node}" if p_node != "none" else ""
    edge_prob = f"acquisition_strategy.tta.p_edge={p_edge}" if p_edge != "none" else ""
    s = (
        f"nohup python main.py model={model} data={dataset} "
        f"acquisition_strategy={strategy} data.num_splits={n_splits} "
        f"model.num_inits={n_inits} print_summary=True "
        f"model.cached=False "
        f"seed={seed} "
        f"acquisition_strategy.adaptation_enabled=False "
        f"acquisition_strategy.tta_enabled=True "
        f"acquisition_strategy.scale={scale} "
        f"acquisition_strategy.tta.strat_node={strat_node} "
        f"acquisition_strategy.tta.strat_edge={strat_edge} "
        f"acquisition_strategy.tta.num={num} "
        f"{node_prob} "
        f"{edge_prob} "
        f"acquisition_strategy.tta.filter={filter} "
        f"acquisition_strategy.tta.probs={probs} "
        f"wandb.name=f{strat_node}_e{strat_edge}_{num}_{f}_{pl}_{p_node}_{p_edge}{seed_text} "
        f"> logs_new/{dataset}_{model}_{strategy}_tta_f{strat_node}_e{strat_edge}_{num}_{f}_{pl}_{p_node}_{p_edge}{seed_text}.log &"
    )
    return s

def generate_prompt_educated_random_by_pred_attribute_notta(dataset, model, strategy, top_percent, low_percent, seed=42):
    strategies = {
        "aleatoric_propagated": ("false", "MAX_SCORE"),
        "MAX_SCORE": ("false", "MAX_SCORE"),
        "entropy": ("true", "ENTROPY"),
        "ENTROPY": ("true", "ENTROPY"),
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    hb, strat = strategies[strategy]
    seed_text = f"_{seed}" if seed != 42 else ""
    
    return (
        f"nohup python main.py model={model} data={dataset} acquisition_strategy=educated_random "
        f"data.num_splits=5 model.num_inits=5 print_summary=True model.cached=True seed={seed} "
        f"acquisition_strategy.adaptation_enabled=False "
        f"acquisition_strategy.scale=1 "
        f"acquisition_strategy.tta_enabled=False "
        f"acquisition_strategy.top_percent={top_percent} acquisition_strategy.low_percent={low_percent} "
        f"+acquisition_strategy.embedded_strategy=acquire_by_prediction_attribute "
        f"+acquisition_strategy.embedded_strategy.higher_is_better={hb} "
        f"+acquisition_strategy.embedded_strategy.attribute={strat} "
        f"+acquisition_strategy.embedded_strategy.propagated=True "
        f"wandb.name={strategy}_{top_percent}_{low_percent}{seed_text} > "
        f"logs_new/{dataset}_{model}_educated_random_{strategy}_{top_percent}_{low_percent}{seed_text}.log &"
    )


def generate_prompt_educated_random_alea_prop(top_percent, low_percent, dataset, model, seed):
   s = (f"nohup python main.py model={model} data={dataset} acquisition_strategy=educated_random "
        f"data.num_splits=5 model.num_inits=5 print_summary=True "
        f"model.cached=False seed={seed} acquisition_strategy.adaptation_enabled=False "
        f"acquisition_strategy.scale=1 "
        f"acquisition_strategy.tta_enabled=True "
        f"acquisition_strategy.tta.strat_node=mask "
        f"acquisition_strategy.tta.strat_edge=mask "
        f"acquisition_strategy.tta.num=100 "
        f"acquisition_strategy.tta.p_node=0.5 "
        f"acquisition_strategy.tta.p_edge=0.4 "
        f"acquisition_strategy.tta.filter=True "
        f"acquisition_strategy.tta.probs=True "
        f"acquisition_strategy.top_percent={top_percent} acquisition_strategy.low_percent={low_percent}  "
        f"+acquisition_strategy.embedded_strategy=acquire_by_prediction_attribute "
        f"+acquisition_strategy.embedded_strategy.higher_is_better=false "
        f"+acquisition_strategy.embedded_strategy.attribute=MAX_SCORE "
        f"+acquisition_strategy.embedded_strategy.propagated=True "
        f"wandb.name=aleatoric_propagated_fmask_emask_100_filter_probs_0.5_0.4_{top_percent}_{low_percent} "
        f"> logs_new/{dataset}_{model}_educated_random_aleatoric_propagated_fmask_emask_100_filter_probs_0.5_0.4_{top_percent}_{low_percent}.log &")
   return s


def augmentation_name(x):
    if x == "fmask":
        return "Feature Mask"
    elif x == "fnoise":
        return "Feature Noise"
    elif x == "emask":
        return "Edge Mask"
    elif x == "fmask,emask":
        return "Feature & Edge Mask"
    elif x == "fnoise,emask":
        return "Feature Noise & Edge Mask"
    return None


def get_count_dict(t, num_nodes = 2810):
    ixs = torch.tensor([l[1:] for l in t["acquired_idxs"]]).flatten()
    count = torch.bincount(ixs, minlength=num_nodes)
    keys = torch.where(count)
    count_dict = {k.item():count[k].item() for k in keys[0]}
    return count_dict, count,ixs

# AGGREGATE GEEM METRICS
def combine_geem_metrics(dataset, strategy = "geem",prefix =None):
    metrics = []
    seed_path = os.path.join("..","output2/runs", dataset, "sgc", strategy)
    filter_fn = lambda x: x.startswith(prefix) if prefix is not None else (len(x.split("_")) == 1)
    for seed in os.listdir(seed_path):
        if (seed != "None") and filter_fn(seed):
            run_path = os.path.join(seed_path, seed)
            for run_dir in os.listdir(run_path):
                if run_dir != "hydra-outputs" and os.path.exists(os.path.join(run_path, run_dir, "acquisition_curve_metrics.pt")):
                    metrics_path = os.path.join(run_path, run_dir, "acquisition_curve_metrics.pt")
                    metric = torch.load(metrics_path, weights_only=True)
                    metrics.append(metric)
    aggregated_metrics = {k:np.array([m[k][0] for m in metrics]) if k != "acquired_idxs" else [m[k][0] for m in metrics]  for k in metrics[0].keys()}
    return aggregated_metrics
    
def aggregate_geem_metrics(dataset, strategy = "geem",prefix = None):
    metrics = combine_geem_metrics(dataset, strategy,prefix)
    metrics["accuracy/test"] *=100
    mean_metrics = {k:np.mean(v, axis=0) if k != "acquired_idxs" else v for k,v in metrics.items()}
    std_metrics = {k:np.std(v, axis=0) if k != "acquired_idxs" else v for k,v in metrics.items()}
    return mean_metrics, std_metrics


def index_function(l):
    match l:
        case 28:
            return [0,5,10,15,20,28]
        case 24:
            return [0,5,10,15,20,24]
        case 12:
            return [0,3,5,7,10,12]
        case 32:
            return [0,5,10,15,20,25,32]
        case _:
            raise ValueError(f"Unknown length: {l}")

def create_df(metrics_dict):
    acs =  {k:v[0]for k,v in metrics_dict.items()}
    df = pd.DataFrame(acs)
    index_list = index_function(len(df)-1)
    df_mean = df.transpose()[index_list]
    stds = {k:v[1]for k,v in metrics_dict.items()}
    df_std = pd.DataFrame(stds).transpose()[index_list]
    df_combined = df_mean.join(df_std, lsuffix="_mean", rsuffix="_std").sort_index(axis=0)
    df_combined["final_acc"] = df_combined[f"{index_list[-1]}_mean"]
    mean_cols = [str(i) + "_mean" for i in index_list]
    std_cols = [str(i) + "_std" for i in index_list]

    for mean_col, std_col in zip(mean_cols, std_cols):
        df_combined[mean_col + "_formatted"] = (df_combined[mean_col]).round(1).astype(str) + " ± " + (df_combined[std_col]).round(1).astype(str)
    df_combined.sort_values(by=f"{index_list[-1]}_mean", ascending=False,inplace=True)
    df_sum = (df.iloc[5:].sum(axis=0) / df.iloc[5:].sum(axis=0).max()).sort_values(ascending=False).to_frame(name="nalc")*100
    df_combined = df_combined.join(df_sum)
    return df_combined, df

def plot_diff(df_to_plot,s = ["age", "anrmab", "entropy", "aleatoric_propagated"] ,o="None", n = "fmask_emask_200_filter_probs_0.5_0.4"):
    
    d = [strategy + "_diff" for strategy in s]
    for strategy in s:
        df_to_plot[strategy + "_diff"] = df_to_plot[strategy + "_" +  n] - df_to_plot[strategy+ "_" + o]
    df_to_plot[d].plot(figsize=(20,10))
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.show()
    
def process_tta(df):
    df["num"] = df.index.map(lambda x: x.split("_")[-5]).astype(int)
    df["p_e"] = df.index.map(lambda x: x.split("_")[-1])
    df["p_e"] = df["p_e"].replace("none", np.nan).astype(float)
    df["p_f"] = df.index.map(lambda x: x.split("_")[-2])
    df["p_f"] = df["p_f"].replace("none", np.nan).astype(float)
    df["m_f"] = df.index.map(lambda x: x.split("_")[-7][1:])
    df["m_e"] = df.index.map(lambda x: x.split("_")[-6][1:])
    df["filter"] = df.index.map(lambda x:  x.split("_")[-4])
    return df


def name_fn(x):
    tta = (" " + x["tta"]) if x["tta"] != "NO" else ""
    filtered = " Filtered" if x["filter"] else ""
    adapted = " Adapted" if x["adapted"] else ""
    return f"{x['strategy']}{tta}{filtered}{adapted} {x['model']}"   
     
def get_count_dict(t, num_nodes = 2810):
    ixs = torch.tensor([l[1:] for l in t["acquired_idxs"]]).flatten()
    count = torch.bincount(ixs, minlength=num_nodes)
    keys = torch.where(count)
    count_dict = {k.item():count[k].item() for k in keys[0]}
    return count_dict, count,ixs

def get_count_dict_binned_stats(t, num_nodes = 2810, num_acquired = 28):
    reshaped_ixs = get_count_dict(t,num_nodes)[2].view(5,5,-1)
    reshaped_ixs.shape
    reshaped_bin_count_split = torch.stack([torch.bincount(reshaped_ixs[i].flatten(), minlength=2810) for i in range(reshaped_ixs.shape[0])])
    reshaped_bin_count_init = torch.stack([torch.bincount(reshaped_ixs[:,i,:].flatten(), minlength=2810) for i in range(reshaped_ixs.shape[1])])
    by_split =((reshaped_bin_count_split > 0).sum()).item()/(5*5*num_acquired)
    by_init = ((reshaped_bin_count_init > 0).sum()).item()/(5*5*num_acquired)
    return by_split, by_init


import matplotlib.pyplot as plt
def hist_plus_mean(tensor, num_bins=30):
    hist_range = (np.min(tensor), np.max(tensor))
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)

    # Compute histograms per row
    hist_counts = np.zeros((28, num_bins))
    for i in range(28):
        hist_counts[i], _ = np.histogram(tensor[i], bins=bin_edges)

    # Compute mean for each row
    row_means = np.mean(tensor, axis=1)
    return hist_counts, row_means, bin_edges
def plot_hist_plus_means(tensor,title,num_bins=30):
    hist_counts, row_means, bin_edges = hist_plus_mean(tensor, num_bins)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(hist_counts.T, aspect='auto', origin='lower',
            extent=[0, 27, bin_edges[0], bin_edges[-1]],
            cmap='coolwarm')

    # Overlay mean line
    x_vals = np.arange(28)
    plt.plot(x_vals, row_means, color='red', linewidth=2)

    # Labels and layout
    plt.colorbar(label='Frequency')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.title(f"{title} histogram and mean" )
    plt.tight_layout()
    plt.show()
    


def plot_hist_plus_means_multiple(hists,row_means,titles,bin_edgesS,vmin, vmax):
    for hist_counts,row_mean,title,bin_edges in zip(hists,row_means,titles,bin_edgesS):
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(hist_counts.T, aspect='auto', origin='lower',
                extent=[0, 27, 0, 1],
                cmap='coolwarm', vmin=vmin, vmax=vmax)

        # Overlay mean line
        x_vals = np.arange(28)
        plt.plot(x_vals, row_mean, color='red', linewidth=2)

        # Labels and layout
        plt.colorbar(label='Frequency')
        plt.xlabel('Iteration')
        plt.ylabel('Probability')
        plt.title(f"{title} histogram and mean" )
        plt.tight_layout()
        plt.show()






def init_split_std(accuracy_array):
    bins = np.array(np.array_split(accuracy_array, len(accuracy_array) // 5))

    # Calculate the standard deviation for each bin
    std_inside_split = np.std(bins, axis=1)
    std_inside_init = np.std(bins, axis=0)
    diff = std_inside_split - std_inside_init
    mean_diff = np.mean(diff)
    return (
        mean_diff,
        np.std(diff),
        np.mean(std_inside_split),
        np.mean(std_inside_init),
        diff,
    )

def compute_stds(df,dataset, all_metrics):
    df["std_diff"] = df.apply(
    lambda x: init_split_std(
        all_metrics[(dataset, x["model"])][x["index_original"]][3]["accuracy/test"]
    )[0],
    axis=1,
    )
    df["std_inside_split"] = df.apply(
    lambda x: init_split_std(
         all_metrics[(dataset, x["model"])][x["index_original"]][3]["accuracy/test"]
    )[2],
    axis=1,
    )
    
    df["std_inside_init"] = df.apply(
    lambda x: init_split_std(
        all_metrics[(dataset, x["model"])][x["index_original"]][3]["accuracy/test"]
    )[3],
    axis=1,
)
    return df