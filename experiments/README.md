# Sweep experiment files

Each file in this directory defines one sweep experiment.  `scripts/submit.sh`
reads the file and submits a single Slurm array job whose size equals the full
Cartesian product of the `sweep:` axes.  `scripts/sweep_runner.py` handles
index→override mapping inside each array task, so the sbatch scripts
(`slurm/d4rl_array.sh`, `slurm/ogbench_array.sh`) stay fully generic and never
need to be edited between experiments.

## Format

```yaml
name: e001_fpot_temp_d4rl              # unique experiment ID; becomes --job-name and +exp= tag
description: human-readable summary    # ignored at runtime, documents intent

env_set: d4rl                          # selects the array runner:  d4rl | ogbench

wandb_project: flowrl-d4rl            # injected as +logging.wandb_project=...
                                       # (omit to fall back to cfg.env.name)

sweep:                                 # Cartesian product axes, declaration order preserved.
  env: [hopper_medium, walker2d_medium]  # outermost loop  (first key)
  agent.w_temperature: [1.0, 2.0]       #   ...
  seed: [2, 4]                          # innermost loop  (last key)

fixed:                                 # appended as overrides to EVERY job (no product)
  agent.q_agg: mean
  agent.use_vabc_td_target: true

slurm:                                 # forwarded verbatim as sbatch CLI flags
  time: "48:00:00"
  mem: "32G"
  cpus_per_task: 4
  partitions: [big_suma_rtx3090, suma_rtx4090]   # comma-joined → --partition
  qos: big_qos
  exclude: [node08, node18]                       # comma-joined → --exclude
```

The iteration order is: first key changes slowest, last key changes fastest.
So for the example above the sequence is:
  idx 0 → env=hopper_medium  w_temperature=1.0  seed=2
  idx 1 → env=hopper_medium  w_temperature=1.0  seed=4
  idx 2 → env=hopper_medium  w_temperature=2.0  seed=2
  ...
  idx 7 → env=walker2d_medium  w_temperature=2.0  seed=4
