from flair.models.iattention import IAttentionHead
import torch
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair import device 

from .config import Config
from . import vq_attention

from .vq_attention import VQbottleneck


class AttentionHead(IAttentionHead):
    def __init__(
        self,
        embeddings,
        tag_dictionary,
        tag_type: str,
        tag_format: str = "BIOES",
        hidden_size: int = 256,
        reproject_embeddings: bool = True,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        loss_weights = None,
        init_from_state_dict: bool = False,
        allow_unk_predictions: bool = False,
        hp=Config()
    ):
        super().__init__(
            embeddings,
            tag_dictionary,
            tag_type,
            tag_format,
            hidden_size,
            reproject_embeddings,
            dropout,
            word_dropout,
            locked_dropout,
            loss_weights,
            init_from_state_dict,
            allow_unk_predictions,
        )

        #self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
        hp.k = len(self.label_dictionary)
        embedding_dim = embeddings.embedding_length
        print("emb_ dimension: ", embedding_dim)
        hp.input_d = embedding_dim
        print("XXXXXXXX:" ,hp.input_d)
        self.model = VQbottleneck(hp)

        #self.model = torch.nn.Linear(hp.input_d, len(self.label_dictionary))
        self.to(device)

    def forward_loss(self, sentences):

        # if there are no sentences, there is no loss
        if len(sentences) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
        sentences = sorted(sentences, key=len, reverse=True)
        gold_labels = self._prepare_label_tensor(sentences)
        sentence_tensor, lengths = self._prepare_tensors(sentences)

        # forward pass to get scores
        scores = self.forward(sentence_tensor, lengths)

        # calculate loss given scores and labels
        return self._calculate_loss(scores, gold_labels)


    def forward(self, sentence_tensor: torch.Tensor, lengths: torch.LongTensor):  # type: ignore[override]
        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # linear map to tag space
        features = self.model(sentence_tensor)

        scores = self._get_scores_from_features(features, lengths)

        return scores

    def evaluate(self, *args, **kwargs):
        print("perplexity: ", self.model.encoder.saved_perplexity)
        print("probs :", torch.argmax(self.model.encoder.saved_probs, dim=-1))
        return super().evaluate(*args, **kwargs)

    def _loss(self, scores, labels):
        #probs = self.model.encoder.saved_probs
        #H = torch.sum(-1*probs*torch.log(probs + 3e-10), dim=-1)
        #H = torch.mean(H)
        return self.loss_function(scores, labels) #- 0.25*H
