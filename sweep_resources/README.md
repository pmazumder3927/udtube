This directory contains a config and a script for running a hyperparameter sweep for UDTube models. 
You'll have to have wandb installed and be logged into it to use this functionality

To begin a sweep, initialize it using wandb sweeps and the sweep_config.yaml:
```commandline
wandb sweep --project <project_name> sweep_config.yaml
```

You can use any yaml, but sweep_config.yaml is left here for convenience.

To run the script:

```commandline
python3 tuning.py --sweep_id <sweep_id> --entity <entity> --project <same project_name from earlier> --count <amt of runs>
```
