"""Credit for a lot of this code:
https://github.com/Kyubyong/nlp_made_easy/blob/master/Pos-tagging%20with%20Bert%20Fine-tuning.ipynb"""

import torch
import transformers
import torch.nn as nn
import lightning.pytorch as pl
import argparse


def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The model that will be used for inference")
    parser.add_argument("--training_input", help="Training data, conllu format")
    parser.add_argument("--predict_input", help="The data we will be predicting on, conllu format")
    parser.add_argument("--predict_output", help="The file we will be writing to, conllu format")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    return parser


class UDTube(pl.LightningModule):
    def __init__(self, model_name, pos_out_size=None, um_out_size=None, lem_vocab_size=None):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        self.bert = transformers.AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.in_feature_size = list(self.bert.parameters())[0].shape[1]
        self.lin_pos = nn.Linear(self.in_feature_size, pos_out_size)
        self.lin_um = nn.Linear(self.in_feature_size, um_out_size)
        self.lin_lem = nn.Linear(self.in_feature_size, lem_vocab_size)

    @staticmethod
    def _calculate_chunks(tokenized):
        """helper to map token ids from the same word to embeddings"""
        chunks = []
        for word_id in tokenized.word_ids():
            if word_id is None:
                # the case of padding [cls], [sep] - don't need
                continue
            start, end = tokenized.word_to_tokens(word_id)
            tokens = slice(start, end)
            if not chunks or chunks[-1] != tokens:
                chunks.append(tokens)
        yield chunks

    def get_word_embs(self, tokenized):
        """get the embeddings for each word (not just subwords)"""
        with torch.no_grad():
            output = self.bert(**tokenized)
            hidden_states = output[2]
        token_embeddings = hidden_states
        ids_and_wembs = []
        for chunk in self._calculate_chunks(tokenized):
            ids_and_wembs.append(
                (tokenized.input_ids[:, chunk],
                 torch.mean(token_embeddings[chunk], dim=1))
            )
        return ids_and_wembs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y_pos, y_um, y_lem = batch
        x = x.view(x.size(0), -1)
        z = self.bert(x)

        x_pos_hat = self.lin_pos(z)
        loss = nn.CrossEntropyLoss(x_pos_hat, x)

        x_um_hat = self.lin_um(z)
        loss += nn.CrossEntropyLoss(x_um_hat, x)

        x_lem_hat = self.lin_lem(z)
        loss += nn.CrossEntropyLoss(x_lem_hat, x)

        self.log("train_loss", loss)
        return loss

    def forward(self, x, y_pos, y_um, y_lem):
        self.bert.eval()
        tokenized = self.tokenizer(x, return_tensors="pt")
        _, embs = zip(*self.get_word_embs(tokenized))

        res = {}
        pos_logits = self.lin_pos(embs)
        y_pos_hat = pos_logits.argmax(-1)
        res['pos'] = {"logits": pos_logits,
                     "gold": y_pos,
                     "pred": y_pos_hat}

        um_logits = self.lin_um(embs)
        y_um_hat = um_logits.argmax(-1)
        res['pos'] = {"logits": um_logits,
                      "gold": y_um,
                      "pred": y_um_hat}

        lem_logits = self.lin_lem(embs)
        y_lem_hat = pos_logits.argmax(-1)
        res['pos'] = {"logits": lem_logits,
                      "gold": y_lem,
                      "pred": y_lem_hat}

        return res


if __name__ == '__main__':
    parser = set_up_parser()
    model = UDTube(parser.model, 2, 2, 2)

