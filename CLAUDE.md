# Cluster Bot - Default Instructions

## Slack notifications
When you finish any task, automatically send a summary to Slack using:

SLACK_WEBHOOK=$(cat /work/log1/erik.mueller/.slack_webhook 2>/dev/null) && curl -X POST -H 'Content-type: application/json' --data '{"text":"YOUR SUMMARY HERE"}' "$SLACK_WEBHOOK"

Include the key results, important numbers, and file paths of any generated plots or output files.
Only skip this if I explicitly tell you "no Slack" in my prompt.

### Plots as images
If the task produced summary plots, also post them as images to the same channel,
#cluster-results (channel ID C0CH0CKQHQC, private, only me as member).
The webhook can only send text, so upload the PNGs with the Slack connector:
slack_get_file_upload_url -> POST the file bytes -> slack_complete_file_upload with
channel_id=C0CH0CKQHQC and a short initial_comment saying what the plot shows.
Post the text summary first, then the plot(s).

## Weights & Biases
Every training run must log to W&B, project **yara-crystals** — plan this in from the start
when writing any new training script, not as an afterthought.
- Log ALL hyperparameters as the run config (model, optimizer, scheduler, loss settings,
  dataset path/dims, split, seed, epochs, git commit, SLURM job/array IDs), per-epoch
  train/val metrics, and final test metrics as the run summary.
- Use a descriptive run name, a `group` per experiment and `tags` (domain, model, variant).
- Setup on this cluster:
  - wandb is not in `.venv`; append `/work/log1/erik.mueller/python-packages` to the END of
    `sys.path` (not PYTHONPATH, it would shadow venv packages).
  - API key: export the `WANDB_API_KEY` line from `/work/log1/erik.mueller/.my_profile` in slurm
    scripts: `eval "$(grep -E '^export WANDB_API_KEY=' /work/log1/erik.mueller/.my_profile)"`
  - `$HOME` is read-only: point WANDB_DIR / WANDB_CACHE_DIR / WANDB_CONFIG_DIR / WANDB_DATA_DIR
    to `/work/log1/erik.mueller/.wandb/...`.
  - Wrap wandb calls so a W&B failure never kills training.
  - Reference implementation: `WandbLogger` in `train_atom_encoding_sweep.py`.
- Disable only if I explicitly say "no wandb".

## General
- Working directory: /work/log1/erik.mueller/code
- Always write progress to a log file when running long jobs
