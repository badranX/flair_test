from flair.trainers import ModelTrainer
from flair.models import SequenceTagger

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings

from .config import Config
from . import vq_attention
from . import flair_model

from flair.datasets import CONLL_03

def create_model_and_corpus():
    corpus = CONLL_03(base_path='/home/banx/dev/data')
    label_type = 'ner'
    label_dict = corpus.make_label_dictionary(label_type=label_type)

# init embedding
    #trans_embedding = TransformerDocumentEmbeddings('bert-base-uncased')

    embedding_types = [
        WordEmbeddings('glove'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    hp = Config()

    model = flair_model.AttentionHead(embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                          )
    #model = SequenceTagger(hidden_size=256,
    #                    embeddings=embeddings,
    #                    tag_dictionary=label_dict,
    #                    tag_type=label_type,
    #                    use_crf=True)

    return model, corpus


def start():
    model, corpus = create_model_and_corpus()
    trainer = ModelTrainer(model, corpus)

    trainer.train('resources/taggers/example-ner',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=200)


def train(model, corpus):
    trainer = ModelTrainer(model, corpus)

    trainer.train('resources/taggers/example-ner',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=30)

if __name__ == "__main__":
    start()
