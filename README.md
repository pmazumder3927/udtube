# UDTube (beta)

UDtube is a trainable neural morphological analyzer built on Pytorch Lightning. 


## Philosophy

Named in homage to the tried-and-true UDPipe, UDTube is built with inference-at-scale in mind. The software avoids loading large amounts of data at any time and instead favors an incremental loading approach to prediction. 
While originally built for the purposes of investigating the linguistic problem of defectivity, UDTube is released for both replication and to enable new and exciting research in morphology.

## Authors

Daniel Yakubov & Kyle Gorman

## Installation

UDTube is built on Python 3.10 and not extensively tested on other Python verisons.

To begin, clone this repo please install the requirements:
```
pip install -r requirements.txt
```

## Usage

Since UDTube is built using the LightningCLI is doesn't hurt to read up on those [docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html). However, straightforward guidance for using UDTube is below:

Training

It is highly preferred that yaml config files are used for any model cycle. These configs can be found under configs/

For fit, there is a default fit_config.yaml

There are A LOT of options in the config, assuming nothing fancy, the most common parameters that might need to be changed depending on your uses are:
- `model.path_name`
- `model.model_name`
- `data.language`
- `data.train_dataset`
- `data.val_dataset`
- `data.batch_size` (depending on your computer)

Once your `fit_config.yaml` is in order, you can run the following:

```
python udtube.py fit --config configs/fit_config.yaml
```

To read more about the options for training, please run
```
python udtube.py fit --help
```

**WandB is required to keep track of runs done with UDTube**

## Testing

If you're evaluating one of your models before using it, the `test_config.yaml` might be of use to you. Like for fitting, some common parameters will need to be set:

- `model.path_name`
- `model.model_name`
- `model.checkpoint`
- `data.checkpoint` (same as `model.checkpoint`, 99% of the time)
- `data.language`
- `data.test_dataset`
- `data.batch_size` (depending on your computer)

The checkpoints will be written to your dir, look for `.ckpt` extensions. Sometimes found hiding in a `lightning_logs/` subdir.

Once your `test_config.yaml` is ready to go:
```
python udtube.py test --config configs/test_config.yaml
```

To read more about the options for testing, please run
```
python udtube.py test --help
```

## Predict

Once you have a model you're ready to throw a boatload of data at, you're ready to predict. The key prediction parameters are:

- `model.path_name`
- `model.model_name`
- `model.checkpoint`
- `data.checkpoint` (same as model.checkpoint, 99% of the time)
- `data.language`
- `data.test_dataset`
- `data.batch_size` (depending on your computer)
- `output_file` (this can be set to stdout, but I recommend setting `trainer.enable_progress_bar` to `false` if doing so)

Once your `predict_config.yaml` is ready to go:
```
python udtube.py predict --config configs/predict_config.yaml
```

To learn more about the params for prediction:
```
python udtube.py predict --help
```

## Contributing
Fork and Pull! 

