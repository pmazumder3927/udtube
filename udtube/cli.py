"""Command-line interface."""

import logging

from pytorch_lightning import cli

from . import callbacks, datamodules, models


class UDTubeCLI(cli.LightningCLI):
    """The UDTube CLI interface.

    Use with `--help` to see full set of options."""

    @staticmethod
    def add_arguments_to_parser(parser: cli.LightningArgumentParser) -> None:
        parser.add_argument(
            "--output_file",
            help="The file to output to, can be set to stdout.",
        )
        # parser.link_arguments("model.model_dir", "data.model_dir")
        # parser.link_arguments("model.model_dir", "trainer.default_root_dir")
        parser.link_arguments("model.encoder", "data.encoder")
        parser.link_arguments("data.reverse_edits", "model.reverse_edits")
        parser.link_arguments(
            "data.pos_classes_cnt",
            "model.pos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.xpos_classes_cnt",
            "model.xpos_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.lemma_classes_cnt",
            "model.lemma_out_label_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_classes_cnt",
            "model.feats_out_label_size",
            apply_on="instantiate",
        )

    def before_instantiate_classes(self) -> None:
        if self.subcommand == "predict":
            self.trainer_defaults["callbacks"] = [
                callbacks.CustomWriter(self.config.predict.output_file)
            ]
        elif self.subcommand == "test":
            self.trainer_defaults["callbacks"] = [
                callbacks.CustomWriter(self.config.test.output_file)
            ]


def main() -> None:
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    UDTubeCLI(models.UDTube, datamodules.DataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
