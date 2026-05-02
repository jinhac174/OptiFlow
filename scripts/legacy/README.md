# Legacy submission scripts

These scripts are kept for reference only and are **superseded** by the
experiment YAML configs + unified submission pipeline:

```
experiments/*.yaml   — sweep / job definitions
scripts/submit.sh    — entrypoint that dispatches via slurm/d4rl_array.sh
                       or slurm/ogbench_array.sh
```

Do not use these scripts for new experiments.
