"""Credit for a lot of this code:
https://github.com/Kyubyong/nlp_made_easy/blob/master/Pos-tagging%20with%20Bert%20Fine-tuning.ipynb"""

import torch
import transformers
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data


class UDTube(nn.Module):
    def __init__(self, model_name, pos_out_size=None, um_out_size=None, lem_vocab_size=None):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
        self.bert = transformers.AutoModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lin_pos = nn.Linear(768, pos_out_size)
        self.lin_um = nn.Linear(768, um_out_size)
        self.lin_lem = nn.Linear(768, lem_vocab_size)

    @staticmethod
    def _inplace_mean(tensors):
        """helper to get inplace mean of tensors"""
        tmp = tensors[0]
        for i in range(1, len(tensors)):
            tmp *= tensors[i]
        avg_tensor = tmp / len(tensors)
        return avg_tensor

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
        return chunks

    def get_word_embs(self, tokenized):
        """get the embeddings for each word (not just subwords)"""
        with torch.no_grad():
            output = self.bert(**tokenized)
            hidden_states = output[2]
        # Swap dimensions 0 and 1.
        # Embeddings are from the last hidden state
        token_embeddings = hidden_states[-1].permute(1, 0, 2)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)  # getting rid of batch (will I regret this?)
        chunks = self._calculate_chunks(tokenized)
        ids_and_wembs = []
        for chunk in chunks:
            ids_and_wembs.append(
                (tokenized.input_ids[:, chunk],
                 self._inplace_mean(token_embeddings[chunk]))
            )
        return ids_and_wembs

    def forward(self, x, y_pos, y_um, y_lem):
        if self.training:
            self.bert.train()
        else:
            self.bert.eval()

        x = x.to(self.device)
        tokenized = self.tokenizer(x, return_tensors="pt")
        _, embs = zip(*self.get_word_embs(tokenized))

        # will have to change the lemma handling here
        pos_logits = self.lin_pos(embs)
        um_logits = self.lin_um(embs)
        lem_logits = self.lin_lem(embs)
        layers = (pos_logits, um_logits, lem_logits)
        names = ('pos', 'um', 'lemmas')
        golds = (y_pos, y_um, y_lem)

        res = dict()
        for name, logits, y in zip(names, layers, golds):
            y = y.to(self.device)
            y_hat = logits.argmax(-1)
            res[name] = {"logits": logits,
                         "gold": y,
                         "pred": y_hat}

        return res


def train(model, data, optimizer):
    model.train()
    prog = tqdm.auto(len(data))
    for i, d in enumerate(data):
        words, x, is_heads, tags, y, seqlens = data
        _y = y  # for monitoring
        optimizer.zero_grad()
        res = model(x, y)
        for task in res:
            logits = task['logits']
            y = task['y']
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            # does this really make sense to do? Take loss of 3 tasks at once?
            loss = nn.CrossEntropyLoss(ignore_index=0)(logits, y)
            loss.backward()
        optimizer.step()
        prog.update(1)

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")


# TODO implement for our purposes
# def eval(model, data):
#     model.eval()
#     Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
#     with torch.no_grad():
#         for i, batch in enumerate(data):
#             words, x, is_heads, tags, y, seqlens = batch
#
#             _, _, y_hat = model(x, y)
#
#             Words.extend(words)
#             Is_heads.extend(is_heads)
#             Tags.extend(tags)
#             Y.extend(y.numpy().tolist())
#             Y_hat.extend(y_hat.cpu().numpy().tolist())
#
#     ## gets results and save
#     with open("result", 'w') as fout:
#         for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
#             y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
#             preds = [idx2tag[hat] for hat in y_hat]
#             assert len(preds) == len(words.split()) == len(tags.split())
#             for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
#                 fout.write("{} {} {}\n".format(w, t, p))
#             fout.write("\n")
#
#     ## calc metric
#     y_true = np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
#     y_pred = np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
#
#     acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)
#
#     print("acc=%.2f" % acc)


if __name__ == '__main__':
    # just an example so i don't forget
    model = UDTube('distilbert_base_uncased', 1, 2, 3)
