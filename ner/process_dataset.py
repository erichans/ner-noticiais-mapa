import jsonlines
from operator import itemgetter
import re
from pathlib import Path
import numpy as np


class ProcessDataset:
    __DATA_DIR = Path('data')

    def __init__(self, dataset_name, debug=False):
        self.dataset_name = dataset_name
        self.debug = debug

    def get_dataset(self):
        token_docs = []
        tag_docs = []

        with jsonlines.open(ProcessDataset.__DATA_DIR / self.dataset_name) as reader:
            for count, obj in enumerate(reader):
                if self.debug and count >= 10:
                    break

                texto = obj['text']
                labels = obj['labels']
                labels = sorted(labels, key=itemgetter(0))
                tokens = re.findall(r"[\w']+|[-.,!?;()]", texto)

                # Ignora aspas
                tokens = [token.replace('\'', '') for token in tokens]
                tokens = [token for token in tokens if len(token) > 0]

                tags = ['O'] * len(tokens)
                indice = 0

                for inicio, fim, label in labels:
                    trecho = texto[inicio:fim]
                    termos_trecho = re.findall(r"[\w']+|[-.,!?;()]", trecho)

                    # Ignora aspas
                    termos_trecho = [termo_trecho.replace('\'', '') for termo_trecho in termos_trecho]
                    termos_trecho = [termo_trecho for termo_trecho in termos_trecho if len(termo_trecho) > 0]

                    inicio_trecho = tokens.index(termos_trecho[0], indice)
                    tags[inicio_trecho] = 'B-' + label

                    if len(termos_trecho) > 1:
                        fim_trecho = tokens.index(termos_trecho[len(termos_trecho) - 1], inicio_trecho)

                        if fim_trecho != inicio_trecho:
                            tags[fim_trecho] = 'I-' + label

                        for i in range(inicio_trecho + 1, fim_trecho):
                            tags[i] = 'I-' + label
                    else:
                        fim_trecho = inicio_trecho

                    indice = max(inicio_trecho, fim_trecho) + 1

                token_docs.append(tokens)
                tag_docs.append(tags)

        return token_docs, tag_docs

    @staticmethod
    def pre_processar_base(textos, tags, tokenizer):
        max_len = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add()
        nova_lista_textos = []
        nova_lista_tags = []

        for i in range(0, len(textos)):
            tokens = textos[i]
            ts = tags[i]
            token_docs, tag_docs = ProcessDataset.pre_processar_tokens(tokens, ts, tokenizer, max_len)
            nova_lista_textos += token_docs
            nova_lista_tags += tag_docs

        return nova_lista_textos, nova_lista_tags

    @staticmethod
    def pre_processar_tokens(tokens, tags, tokenizer, max_len):
        """
        Pré-processa uma sequência de tokens, quebrando conjuntos com quantidade de tokens superior a um limite especificado
        em conjuntos de sequências menores.
        :param tokens A sequência original de tokens.
        :param tags A sequência de tags atribuídas à sequência de tokens (previamente definidas de forma supervisionada).
        :param tokenizer O tokenizador em uso.  Será usado para contabilizar os tokens a serem gerados pelo tonkenizador,
        uma vez que essa quantidade pode superar o limite pré-estabelecido pelo modelo a ser treinado.
        :param max_len O limite máximo de tokens estabelecido pelo modelo a ser treinado/utilizado.
        :return As subsequências de tokens e tags obtidos pela quebra, na forma de uma tupla de listas
        (token_docs, tag_docs).
        """
        subword_len_counter = 0
        indices_sublistas = [0]
        token_docs = []
        tag_docs = []

        for i, token in enumerate(tokens):
            current_subwords_len = len(tokenizer.tokenize(token))

            # Filtra caracteres especiais
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) >= max_len:
                indices_sublistas.append(i)
                subword_len_counter = current_subwords_len
            else:
                subword_len_counter += current_subwords_len

        for i in range(0, len(indices_sublistas)):
            if i + 1 < len(indices_sublistas):
                token_docs.append(tokens[indices_sublistas[i]:indices_sublistas[i + 1]])
                tag_docs.append(tags[indices_sublistas[i]:indices_sublistas[i + 1]])
            else:
                token_docs.append(tokens[indices_sublistas[i]:])
                tag_docs.append(tags[indices_sublistas[i]:])

        return token_docs, tag_docs

    @staticmethod
    def encode_tags(tags, encodings, tag2id):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []

        assert len(encodings.offset_mapping) == len(labels)
        for i, (doc_labels, doc_offset) in enumerate(zip(labels, encodings.offset_mapping)):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=np.int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            assert len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)]) == len(doc_labels)
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels
