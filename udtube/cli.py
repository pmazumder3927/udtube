"""Command-line interface."""

import logging

from lightning.pytorch import cli

from . import data, models


class UDTubeCLI(cli.LightningCLI):
    """The UDTube CLI interface.

    Use with `--help` to see full set of options."""

    @staticmethod
    def add_arguments_to_parser(parser: cli.LightningArgumentParser) -> None:
        # Passed to the custom writer callback.
        parser.add_argument("--output_file", help="Path for output file")
        # Links.
        parser.link_arguments("model.encoder", "data.encoder")
        parser.link_arguments("data.model_dir", "trainer.default_root_dir")
        parser.link_arguments(
            "data.upos_tagset_size",
            "model.upos_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.xpos_tagset_size",
            "model.xpos_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.lemma_tagset_size",
            "model.lemma_out_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.feats_tagset_size",
            "model.feats_out_size",
            apply_on="instantiate",
        )


def main() -> None:
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    UDTubeCLI(models.UDTube, data.DataModule, save_config_callback=None)


if __name__ == "__main__":
    main()
