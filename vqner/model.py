from flair.embeddings import WordEmbeddings, StackedEmbeddings

from .config import Config
from . import vq_attention
from . import flair_model

from flair.datasets import CONLL_03


def create_model_and_corpus():
    corpus = CONLL_03(base_path='/home/banx/dev/data')
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    embedding_types = [
        WordEmbeddings('glove')
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    hp = Config()
    large_head = vq_attention.VQbottleneck(hp)

    model = flair_model.AttentionHead(embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                          )

    return model, corpus
