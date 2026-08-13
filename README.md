# MARL Drug Design

Code for the publication *"In silico drug design with multi agent reinforcement learning"*.

Datasets used by the training pipeline are in [data/](data/). Ready-to-use training configurations are in [configs/](configs/).

## Contents
- [Environment setup](#environment-setup)
- [Training entry point](#training-entry-point)
- [ADME property endpoints (not included)](#adme-property-endpoints-not-included)
- [Step 1 — Pre-training individual (single-property) agents](#step-1--pre-training-individual-single-property-agents)
- [Step 1.5 — Combining single agents into a composite checkpoint](#step-15--combining-single-agents-into-a-composite-checkpoint)
- [Step 2 — Training composite (multi-agent) models](#step-2--training-composite-multi-agent-models)
- [End-to-end example pipeline (bash)](#end-to-end-example-pipeline-bash)
- [Step 3 — Extracting target molecules from a completed run](#step-3--extracting-target-molecules-from-a-completed-run)
- [Outputs](#outputs)

## Environment setup

Create the conda environment from the provided spec:

```bash
conda env create -f environment.yml
conda activate marlenv
```

## Training entry point

All training runs are launched through [train.py](train.py) with a single `--config` argument pointing at a JSON file:

```bash
python train.py --config <path/to/config.json>
```

The pipeline (see [pipeline/training_runner.py](pipeline/training_runner.py) and [pipeline/data_pipeline.py](pipeline/data_pipeline.py)) will:
1. Load and fragment the datasets referenced by the config (`deepfmpo.train_file`, `deepfmpo.eval_file`, `deepfmpo.lead_file`).
2. Start the scoring REST API automatically when any property `types` require it (e.g. `graphpredict`).
3. Train the agent(s) and periodically evaluate.
4. Save models under the path indicated by `algorithm.pre_train_path` and logs under `logdir`.

## ADME property endpoints (not included)

The ADME single-agent configs (`single_agent_caco_strict.json`, `single_agent_herg_strict.json`, `single_agent_hlm_strict.json`, `single_agent_cyp_strict.json`, `single_agent_logD_strict.json`) and every `*_adme5` composite config expect a scoring REST API that returns predictions for hERG, logD, Caco-2, HLM clearance and CYP3A4. **The ADME prediction models themselves are not shipped with this repository due to propreitary reasons** — the corresponding property `types` in those configs are set to `graphpredict`, meaning they will call out to the scoring REST API started in Step 3 of [train.py](train.py). The same applies for the Factor Xa model.

To use the ADME configs you must plug in your own endpoints, either by:
- Providing your own scoring service that responds on `http://localhost:2000` with the property names used in the config (see `utils/train_utils.py::start_rest_uwsgi` and `wait_for_rest_api`), or
- Editing the composite / single-agent config so `properties.types` uses a scorer you do provide.

The RDKit-only configs (logP, MolWeight, TPSA, QED) do not need any external endpoint and are the recommended starting point.

## Step 1 — Pre-training individual (single-property) agents

Each composite model is built from agents that are first pre-trained to optimise a single property. The single-property configs live in [configs/marl_configs/single_agent_props/](configs/marl_configs/single_agent_props/):

| Config | Property optimised | Needs external endpoint? |
|---|---|---|
| [single_agent_ic50_fxa.json](configs/marl_configs/single_agent_props/single_agent_ic50_fxa.json) | FXa pIC50 (target binding) | Yes |
| [single_agent_logp_strict.json](configs/marl_configs/single_agent_props/single_agent_logp_strict.json) | logP | No (RDKit) |
| [single_agent_molweight_strict.json](configs/marl_configs/single_agent_props/single_agent_molweight_strict.json) | Molecular weight | No (RDKit) |
| [single_agent_tpsa_strict.json](configs/marl_configs/single_agent_props/single_agent_tpsa_strict.json) | TPSA | No (RDKit) |
| [single_agent_qed_strict.json](configs/marl_configs/single_agent_props/single_agent_qed_strict.json) | QED | No (RDKit) |
| `single_agent_{logD,caco,herg,hlm,cyp}_strict.json` | ADME properties | Yes — see [ADME property endpoints](#adme-property-endpoints-not-included) |

Run any one of them, e.g.:

```bash
# RDKit property agents (needed for the *_rdkit composite models)
python train.py --config configs/marl_configs/single_agent_props/single_agent_logp_strict.json
python train.py --config configs/marl_configs/single_agent_props/single_agent_molweight_strict.json
python train.py --config configs/marl_configs/single_agent_props/single_agent_tpsa_strict.json
```

Each single-agent config uses `n_agents: 1`, `pre_train: false`, and writes a checkpoint under `ia2c_models/t_{...}_{config_name}/`. Repeat for whichever agents you need for the composite model you plan to train next.

## Step 1.5 — Combining single agents into a composite checkpoint

The composite trainer loads a single `models.pt` file that already contains all N agents (see `IAA2C.restore` in [marl_algorithms/a2c/iaa2c_v1.py](marl_algorithms/a2c/iaa2c_v1.py)). The per-agent checkpoints produced in Step 1 must therefore be merged into one composite checkpoint before running any `combined_w_pretrained_*` config. This is done with [utils/combine_single_agents.py](utils/combine_single_agents.py), which:

1. Loads each single-agent `models.pt`.
2. Expands the first-layer input weights from 1-agent to N-agent observation shape (mol features are copied; privileged features are zero-initialised).
3. Maps each source agent to the correct target slot by matching `properties.names` between the single-agent config and the target composite config.
4. Writes `models.pt` + `composite_metadata.json` (with `pretrained_agent_indices`) to the output directory, which must match `algorithm.pre_train_path` in the composite config.

Example — build the `combined_mpo_rdkit` checkpoint expected by every `*_rdkit` composite config:

```bash
python utils/combine_single_agents.py \
    --source_models \
        ia2c_models/single_agent_molweight_strict \
        ia2c_models/single_agent_tpsa_strict \
        ia2c_models/single_agent_logp_strict \
    --source_configs \
        configs/marl_configs/single_agent_props/single_agent_molweight_strict.json \
        configs/marl_configs/single_agent_props/single_agent_tpsa_strict.json \
        configs/marl_configs/single_agent_props/single_agent_logp_strict.json \
    --target_config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json \
    --output ia2c_models/combined_mpo_rdkit
```

Notes:
- The order of `--source_models` / `--source_configs` does not have to match the target order — the script matches by property name.
- Target properties without a matching source are randomly initialised and receive a higher LR during fine-tuning via `composite_metadata.json`.
- The output directory must be the exact `pre_train_path` set in the composite config you plan to run.
- The same `combined_mpo_rdkit` checkpoint is reused across the FXa, D4 and Renin `*_rdkit` composite configs — you only need to build it once.
- The equivalent `combined_mpo_adme5` checkpoint is built the same way but requires the ADME endpoints described above.

## Step 2 — Training composite (multi-agent) models

Composite configurations combine multiple property agents into a single MARL run. They live in [configs/marl_configs/](configs/marl_configs/), organised by target and property set:

- FXa target: [fxa_rdkit/](configs/marl_configs/fxa_rdkit/), [fxa_adme5/](configs/marl_configs/fxa_adme5/)
- Dopamine D4 target: [d4_rdkit/](configs/marl_configs/d4_rdkit/), [d4_adme5/](configs/marl_configs/d4_adme5/)
- Renin target: [renin_rdkit/](configs/marl_configs/renin_rdkit/), [renin_adme5/](configs/marl_configs/renin_adme5/)

The `*_adme5` folders require the external ADME endpoints (see the note above); the `*_rdkit` folders work out of the box.

Each folder contains three variants of the same composite:

| Filename pattern | Behaviour |
|---|---|
| `combined_no_pretrain_*.json` | Train the composite from scratch (`pre_train: false`). Steps 1 and 1.5 are not required. |
| `combined_w_pretrained_*.json` | Load the composite checkpoint assembled in Step 1.5 (`pre_train: true`, `pre_train_path` points at the merged `models.pt`) and fine-tune jointly. |
| `combined_w_pretrained_*_100k_*.json` | Same as above but with a shorter (~100k step) fine-tuning schedule. |

Run a composite exactly like a single-agent run:

```bash
#  3 RDKit properties, from scratch on the fxa dataset
python train.py --config configs/marl_configs/fxa_rdkit/combined_no_pretrain_fxa_logp_molweight_tpsa.json

# FXa + 3 RDKit properties, initialised from the pre-trained single agents
python train.py --config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json
```

Swap `fxa` for `d4` or `renin` (and the matching folder) to train composites for the other targets.

For `combined_w_pretrained_*` configs to work, the merged composite checkpoint produced in Step 1.5 must exist at the path given by `algorithm.pre_train_path` in that config. If you write the merged checkpoint elsewhere, update `pre_train_path` accordingly before launching the composite run.

## End-to-end example pipeline (bash)

[bash_scripts/train_single_pipeline.sh](bash_scripts/train_single_pipeline.sh) runs the full three-phase pipeline end-to-end for the 3 RDKit properties composite for the Fxa dataset:

1. **Phase 1** — trains the three single-property agents (`molweight`, `tpsa`, `logp`) via `python train.py --config ...`.
2. **Phase 2** — derives each single-agent model directory from its config and calls `utils/combine_single_agents.py` to build `ia2c_models/combined_mpo_rdkit`.
3. **Phase 3** — fine-tunes the composite by running `python train.py --config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json`.

Run it from the repo root:

```bash
bash bash_scripts/train_single_pipeline.sh
```

Adapt `SINGLE_CONFIGS`, `COMBINED_CONFIG` and `COMBINED_MODEL_DIR` at the top of the script to reproduce the D4 or Renin `*_rdkit` composites, or to switch to `*_adme5` once you have connected your ADME endpoints.

## Step 3 — Extracting target molecules from a completed run

Every training run writes per-agent CSV logs to `ia2c_results/<config_name>/` (training) and `ia2c_results/<config_name>_eval/` (evaluation), where `<config_name>` is the JSON filename without extension. Each folder contains one `agent_0.csv`, `agent_1.csv`, … per property agent, with columns `Step`, `Episode_number`, `Start_Smiles`, `End_Smiles`, `Target`, `MPO_Score`, `Scores` and the reward components. See `save_dflist_to_csv` in [utils/proc_utils.py](utils/proc_utils.py) for the writer.

The scripts in [plot_utils/](plot_utils/) turn those raw agent logs into the target-molecule tables you want. The single entry point is [plot_utils/marl_component_plots.py](plot_utils/marl_component_plots.py), which auto-discovers `agent_*.csv` in `--data_dir` and produces both the reward-component plots and the deduplicated target/non-target CSVs.

Example — extract targets from the FXa + 3 RDKit composite run:

```bash
python plot_utils/marl_component_plots.py \
    --data_dir ia2c_results/combined_w_pretrained_fxa_logp_molweight_tpsa \
    --save_dir plots/combined_w_pretrained_fxa_logp_molweight_tpsa \
    --config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json
```

Repeat with `--data_dir ia2c_results/<run_name>_eval` to get the equivalent files for the evaluation logs. Point `--data_dir` at whichever `ia2c_results/<config_name>` folder the run produced.

Relevant files written into `--save_dir`:

| File | Contents |
|---|---|
| `unique_target_molecules.csv` | Deduped `End_Smiles` for rows where `Target == True`, with `SMILES`, `MPO_Score`, `Quality_Score`. Produced by `compute_unique_molecules_over_steps` in [plot_utils/marl_plot_utils.py](plot_utils/marl_plot_utils.py). |
| `unique_non_target_molecules.csv` | Same, but for rows where `Target == False`. |
| `molecules_with_properties.csv` | Every target molecule with the `Scores` list column parsed into one column per property (property names come from `--config`). Produced by `extract_molecule_properties` in [plot_utils/marl_plot_utils.py](plot_utils/marl_plot_utils.py). |

Optional follow-up analyses (all read the CSVs above):

```bash
# Property distributions vs. leads for the targets found in the run
python plot_utils/property_analysis.py \
    --csv plots/combined_w_pretrained_fxa_logp_molweight_tpsa/molecules_with_properties.csv \
    --config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json \
    --save_dir plots/combined_w_pretrained_fxa_logp_molweight_tpsa/property_analysis

# MPO / quality-score sanity checks against a reference set
python plot_utils/check_mpo.py \
    --csv plots/combined_w_pretrained_fxa_logp_molweight_tpsa/unique_target_molecules.csv \
    --config configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json

# Compare produced targets against approved / reference molecules
python plot_utils/find_common_mols.py \
    --csv1 plots/combined_w_pretrained_fxa_logp_molweight_tpsa/unique_target_molecules.csv \
    --csv2 <other_run>/unique_target_molecules.csv \
    --approved data/master_data_all_plus_approved.csv \
    --output_dir plots/combined_w_pretrained_fxa_logp_molweight_tpsa/comparison
```

## Outputs

- Model checkpoints: written under `algorithm.pre_train_path` (composite configs) or the run's model directory.
- Training / evaluation logs: written under `logdir` (default `logging/`) by the `FileSystemLogger`.
- Per-agent step-by-step logs: `ia2c_results/<config_name>/agent_{i}.csv` (training) and `ia2c_results/<config_name>_eval/agent_{i}.csv` (evaluation).
- Target-molecule CSVs (`unique_target_molecules.csv`, `unique_non_target_molecules.csv`, `molecules_with_properties.csv`): produced by [plot_utils/marl_component_plots.py](plot_utils/marl_component_plots.py) as described in Step 3.
