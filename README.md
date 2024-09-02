# UDTube (beta)

UDTube is a neural morphological analyzer based on
[PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/), and
[Hugging Face transformers](https://huggingface.co/docs/transformers/en/index).

## Philosophy

Named in homage to the venerable
[UDPipe](https://lindat.mff.cuni.cz/services/udpipe/), UDTube is focused on
incremental inference, allowing it to be used to label large text collections.

## Design

The UDTube model consists of a pre-trained (and possibly, fine-tuned)
transformer encoder which feeds into a classifier layer with many as four heads
handling the different morphological tasks.

Lightning is used to generate the [training, validation, inference, and
evaluation
loops](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks).
The [LightningCLI
interface](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)
is used to provide a user interface and manage configuration.

Below, we use [YAML](https://yaml.org/) to specify configuration options, and we
strongly recommend users do the same. However, most configuration options can
also be specified using POSIX-style command-line flags.

## Installation

To install UDTube and its dependencies, run the following command:

    pip install .

## File formats

Other than YAML configuration files, most operations use files in
[CoNLL-U](https://universaldependencies.org/format.html) format. This is a
10-column tab-separated format with a blank line between each sentence and `#`
used for comments. In all cases, the `ID` and `FORM` field must be fully
populated; the `_` blank tag can be used for unknown fields.

Many of our experiments are performed using CoNLL-U data from the [Universal
Dependencies project](https://universaldependencies.org/).

## Tasks

UDTube can perform up to four morphological tasks simultaneously:

-   Lemmatization is performed using the `LEMMA` field and [edit
    scripts](https://aclanthology.org/P14-2111/).

-   [Universal part-of-speech
    tagging](https://universaldependencies.org/u/pos/index.html) is performed
    using the `UPOS` field: enable with `data: use_upos: true`.

-   Language-specific part-of-speech tagging is performed using the `XPOS`
    field: enable with `data: use_xpos: true`.

-   Morphological feature tagging is performed using the `FEATS` field: enable
    with `data: use_feats: true`.

The following caveats apply:

-   Note that many newer Universal Dependencies datasets do not have
    language-specific part-of-speech-tags.
-   The `FEATS` field is treated as a single unit and is not segmented in any
    way.
-   One can convert from [Universal Dependencies morphological
    features](https://universaldependencies.org/u/feat/index.html) to [UniMorph
    features](https://unimorph.github.io/schema/) using
    [`scripts/convert_to_um.py`](scripts/convert_to_um.py).
-   UDTube does not perform dependency parsing at present, so the `HEAD`,
    `DEPREL`, and `DEPS` fields are ignored and should be specified as `_`.

## Usage

The `udtube` command-line tool uses a subcommand interface, with the four
following modes.

### Training (`fit`)

In `fit` mode, one trains a UDTube model from scratch. Naturally, most
configuration options need to be set at training time. E.g., it is not possible
to switch between different pre-trained encoders or enable new tasks after
training.

#### Seeding

Setting the `seed_everything:` argument to some value ensures a reproducible
experiment.

#### Encoder

The encoder layer consists of a pre-trained BERT-style transformer model. By
default, UDTube uses multilingual cased BERT
(`model: encoder: google-bert/bert-base-multilingual-cased`). In theory, UDTube
can use any Hugging Face pre-trained encoder so long as it provides a
`AutoTokenizer` and has been exposed to the target language. We [list all the
Hugging Face encoders we have tested thus far](udtube/encoders.py), and warn
users when selecting an untested encoder. Since there is no standard for
referring to the between-layer dropout probability parameter, it is in some
cases also necessary to specify what this argument is called for a given model.
We welcome pull requests from users who successfully make use of encoders not
listed here.

So-called "tokenizer-free" pre-trained encoders like ByT5 are not currently
supported as they lack an `AutoTokenizer`.

#### Classifier

The classifier layer contains up to four sequential linear heads for the four
tasks described above. By default all four are enabled, as per the following
YAML snippet:

    data:
      use_lemma: true
      use_upos: true
      use_xpos: true
      use_feats: true

If, for example, the `XPOS` field is not set, one should disable it by setting
`use_xpos: false`.

#### Optimization

UDTube uses separate optimizers and LR schedulers for the encoder and
classifier. The intuition behind this is that we may wish to make slow, small
changes (or possibly, no changes at all) to the pre-trained encoder, whereas we
wish to make more rapid and larger changes to the classifier.

The following YAML snippet shows a simple configuration that encapsulates this
principle. It uses the Adam optimizer for both encoder and classifier, but uses
a lower learning rate for the encoder with a linear warm-up and a higher
learning rate for the classifier.

    model:
      encoder_optimizer:
        class_path: torch.optim.Adam
        init_args:
          lr: 1e-5
      encoder_scheduler:
        class_path: udtube.schedulers.WarmupInverseSquareRoot
        init_args:
          warmup_epochs: 5
      classifier_optimizer:
        class_path: torch.optim.Adam
        init_args:
          lr: 1e-3
      classifier_scheduler:
        class_path: lightning.pytorch.cli.ReduceLROnPlateau
        init_args:
          monitor: val_loss
          factor: 0.1
      ...

To use a fixed learning rate for one or the other layer, specify
`class_path: udtube.schedulers.DummyScheduler`.

#### Callbacks

This mode automatically uses the
[`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
callback to handle checkpoint generation. One may wish to add additional
callbacks like the following:

...FIXME

#### Logging

By default, this mode performs minimal logging to standard error. One may wish
to add additional loggers like the following:

...FIXME

#### Other options

Dropout probability is specified using `model: dropout: ...` and defaults to
0.5.

The encoder has multiple layers. The input to the classifier consists of just
the last few layers mean-pooled together. The number of layers used for
mean-pooling is specified using `model: pooling_layers: ...` and defaults to 4.

By default, lemmatization uses reverse-edit scripts. This is appropriate for
predominantly suffixal languages, which are thought to represent the majority of
the world's languages. If working with a predominantly prefixal language,
disable this with `model: reverse_edits: false`.

Batch size is specified using `data: batch_size: ...` and defaults to 32.

There are a number of ways to specify how long a model should train for. For
example, the following YAML snippet;

    trainer:
      max_epochs: 100
      max_time: 00:06:00:00
      ...

specifies that training should run for 100 epochs or 6 wall-clock hours,
whichever comes first.

### Validation (`validate`)

In `validation` mode, one runs the validation step over labeled validation data
(specified as `data: val: path/to/validation.conllu`) using a previously trained
model checkpoint (`model: ckpt_path: path/to/checkpoint.ckpt`), recording total
loss and per-task accuracies. In practice this is mostly useful for debugging.

### Inference (`predict`)

In `predict` mode, a previously trained model checkpoint
(`model: ckpt_path: path/to/checkpoint.ckpt`) is used to label a CoNLL-U file.
In addition to the model checkpoint, one must provide the path to the input file
(`data: predict: path/to/predict.conllu`) and a path for the labeled output file
(??).

This mode depends on a custom callback which converts model logits into CoNLL-U
files.

The following caveats apply:

-   In `predict mode` UDTube loads the file to be labeled incrementally (i.e.,
    one sentence at a time) so this can be used with very large files.
-   If the input file already has any of the four task fields (`LEMMA`, `UPOS`,
    `XPOS`, or `FEATS`) specified, they will be overwritten. In this mode,
-   Use [`scripts/pretokenize.py`](scripts/pretokenize.py) to convert raw text
    files to CoNLL-U input files.

## Additional scripts

See [`scripts/README.md`](scripts/README.md) for details on provided scripts not
mention above.

## License

UDTube is distributed under an [Apache 2.0 license](LICENSE.txt).

## Contribution

We welcome contributions using the fork-and-pull model.

## References

If you use UDTube in your research, we would appreciate it if you cited the
following document, which describes the model:

Yakubov, D. 2024. [How do we learn what we cannot
say?](https://academicworks.cuny.edu/gc_etds/5622/) Master's thesis, CUNY
Graduate Center.
