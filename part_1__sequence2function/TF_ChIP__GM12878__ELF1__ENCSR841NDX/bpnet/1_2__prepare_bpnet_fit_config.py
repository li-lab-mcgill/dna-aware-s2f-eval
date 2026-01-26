"""
Sample format

{
	"n_filters": 64,
	"n_layers": 8,
	"profile_output_bias": true,
	"count_output_bias": true,
	"name": "example",
	"batch_size": 64,
	"in_window": 2114,
	"out_window": 1000,
	"max_jitter": 128,
	"reverse_complement": true,
	"reverse_complement_average": true,
	"max_epochs": 50,
	"validation_iter": 100,
	"early_stopping": 10,
	"negative_ratio": 0.33,
	"lr": 0.001,
	"count_weight_loss": null,
	"dtype": "float32",
	"scheduler": true,
	"device": "cuda",
	"verbose": false,

	"min_counts": 0,
	"max_counts": 99999999,

	"training_chroms": ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", 
	  "chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", 
	  "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
	"validation_chroms": ["chr8", "chr10"],

	"sequences":"hg38.fa",
	
	"loci":[
		"peaks.bed.gz"
	]
	"negatives":[
		"negatives.bed.gz
	],
	"exclusion_lists":[
		"hg38.blacklist.bed.gz"
	],
	"signals":[
	  "input.plus.bigWig", 
	  "input.minus.bigWig"
	],
	"controls":[
	  "control.plus.bigWig", 
	  "control.minus.bigWig"
	],
	
	"random_state": 0
}
"""

import json
import os
import sys
import os.path as osp


# Add project root to path
print(os.path.dirname(__file__))
# for gloabl utils and config import
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
from config import *

genomes_dir = osp.join(
	MANUSCRIPT_EXPERIMENTS_DIR,
    "section_1__ConditionalLM_as_diagnostic_probe/workspace/data/genomes/hg38"
)
data_dir = osp.join(
	MANUSCRIPT_EXPERIMENTS_DIR,
    "section_1__ConditionalLM_as_diagnostic_probe/workspace/data",
    "TF_ChIP/GM12878__ELF1__ENCSR841NDX",
    "preprocessed"
)

output_dir = osp.join(
	MANUSCRIPT_EXPERIMENTS_DIR,
	"section_1__ConditionalLM_as_diagnostic_probe/workspace/runs/",
	"TF_ChIP/GM12878__ELF1__ENCSR841NDX/S2F/bpnetlite__without_control_track_input"
)
os.makedirs(output_dir, exist_ok=True)

# load default bpnet fit 
filepath = osp.join(output_dir, "defaults/bpnet_fit_default.json")
with open(filepath, 'r') as f:
	bpnet_fit_default = json.load(f)

# Hardcoded variables
splits_dir = osp.join(genomes_dir, "splits")
signal_files = [
    osp.join(data_dir, "main_experiment.plus.bw"),
    osp.join(data_dir, "main_experiment.minus.bw"),
]
control_files = [
    osp.join(data_dir, "control_experiment.plus.bw"),
    osp.join(data_dir, "control_experiment.minus.bw"),
]
loci_files = [
    osp.join(data_dir, "peaks.bed"),
]
negative_loci_filepath_template = \
    osp.join(data_dir, "nonpeaks.{}.bed")

output_path_template = \
    osp.join(output_dir, "{}", "fit_config.json")

ref_genome_filepath = osp.join(genomes_dir, "hg38.fa")
max_jitter = 100


# iterate over fold names
fold_names = [f"fold_{idx}" for idx in range(5)] 
for fold_name in fold_names:
    # Get split files for the current fold
    filepath = osp.join(splits_dir, f"{fold_name}.json")
    with open(filepath, 'r') as f:
        split_data = json.load(f)
        
    # copy the default bpnet fit and update it
    bpnet_fit_without_controls = bpnet_fit_default.copy()
	# update the json 
    bpnet_fit_without_controls.update({
		"name": f"bpnet_without_controls__fit__{fold_name}",
		"training_chroms": split_data["train"],
		"validation_chroms": split_data["valid"],
		"sequences": ref_genome_filepath,
        "reverse_complement": True,
		"reverse_complement_average": False,
        "max_jitter": max_jitter,
		"loci": loci_files,
		"negatives": [negative_loci_filepath_template.format(fold_name)],
		"signals": signal_files,
		"controls": None,
	})
    # save config
    filepath = output_path_template.format(fold_name)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
          json.dump(bpnet_fit_without_controls, f, indent=4)
          