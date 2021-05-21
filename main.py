import torch
import transformers.utils.logging

from ner.process_dataset import ProcessDataset

from transformers import BertTokenizerFast, BertForTokenClassification, \
    Trainer, TrainingArguments, EvalPrediction, IntervalStrategy

from ner.model import BertNER
from ner.model_crf import BertCRF

from sklearn.model_selection import train_test_split
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    classification_report, performance_measure
from typing import Tuple, List, Dict

from torch import nn
from ner.news_dataset import NewsDataset

from sklearn_crfsuite import metrics

import logging
import pickle
from pathlib import Path


def run():
    logging.basicConfig(level=logging.INFO)
    transformers.logging.set_verbosity_info()

    tokenizer = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased', model_max_length=512,
                                                  do_lower_case=False)

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', model_max_length=512,
    #                                               do_lower_case=False)

    train_texts, val_texts, train_tags, val_tags = get_data(tokenizer)

    unique_tags = sorted(set(tag for doc in train_tags + val_tags for tag in doc))

    tag2id = {tag: id_gerado for id_gerado, tag in enumerate(unique_tags)}
    id2tag = {id_geraddo: tag for tag, id_geraddo in tag2id.items()}

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                return_tensors='pt')
    eval_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                               return_tensors='pt')

    train_labels = torch.LongTensor(ProcessDataset.encode_tags(train_tags, train_encodings, tag2id))
    eval_labels = torch.LongTensor(ProcessDataset.encode_tags(val_tags, eval_encodings, tag2id))

    train_encodings.pop("offset_mapping")
    eval_encodings.pop("offset_mapping")

    train_dataset = NewsDataset(train_encodings, train_labels)
    eval_dataset = NewsDataset(eval_encodings, eval_labels)

    # model = BertNER.from_pretrained('neuralmind/bert-base-portuguese-cased', id2label=id2tag, label2id=tag2id)
    # model = BertNER.from_pretrained('neuralmind/bert-large-portuguese-cased', id2label=id2tag, label2id=tag2id)
    model = BertCRF.from_pretrained('neuralmind/bert-base-portuguese-cased', id2label=id2tag, label2id=tag2id)
    # model = BertCRF.from_pretrained('neuralmind/bert-large-portuguese-cased', id2label=id2tag, label2id=tag2id)

    # model = BertNER.from_pretrained('bert-base-multilingual-cased', id2label=id2tag, label2id=tag2id)
    # model = BertCRF.from_pretrained('bert-base-multilingual-cased', id2label=id2tag, label2id=tag2id)

    # model = BertForTokenClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', id2label=id2tag,
    #                                                    label2id=tag2id)

    BATCH_SIZE = 3
    train_epochs = 30
    experiment_name = f'{train_epochs}_epochs_base_pt_BR_crf'

    training_args = TrainingArguments(
        output_dir=f'./results/pub/{experiment_name}',  # output directory
        num_train_epochs=train_epochs,  # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,  # batch size for evaluation
        # learning_rate=3e-05,    # default:  5e-05,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.EPOCH,
        metric_for_best_model='f1_masked',
        load_best_model_at_end=True
    )

    def mask_output_iobe(batch_predictions: np.ndarray, j: int, batch_predictions_tags: List):
        # First token: possible predictions: O, B-*. Others = -100 or
        # if last prediction was E-XXX, then only accepts B-XYZ or O or
        # if last prediction was O, then only accepts B-XYZ or O
        if len(batch_predictions_tags) == 0 or batch_predictions_tags[-1][:1] in ['E', 'O']:
            batch_predictions[j, [id for id, tag in id2tag.items() if tag[:1] not in ['B', 'O']]] = -100
        # if last prediction was B-XXX, then only accepts I-XXX, E-XXX, O or B-XYZ
        elif batch_predictions_tags[-1][:1] == 'B':
            batch_predictions[j, [id for id, tag in id2tag.items() if tag not in [f'I-{batch_predictions_tags[-1][2:]}',
                                                                                  f'E-{batch_predictions_tags[-1][2:]}',
                                                                                  'O'] and tag[:1] != 'B']] = -100
        # if last prediction was I-XXX, then only accepts I-XXX, E-XXX
        elif batch_predictions_tags[-1][:1] == 'I':
            batch_predictions[j, [id for id, tag in id2tag.items() if tag not in [f'I-{batch_predictions_tags[-1][2:]}',
                                                                                  f'E-{batch_predictions_tags[-1][2:]}']]] = -100

        return id2tag[np.argmax(batch_predictions[j])]

    def mask_output_iob2(batch_predictions: np.ndarray, j: int, batch_predictions_tags: List):
        # First token: possible predictions: O, B-*. Others = -100 or
        # if last prediction was E-XXX, then only accepts B-XYZ or O or
        # if last prediction was O, then only accepts B-XYZ or O
        if len(batch_predictions_tags) == 0 or batch_predictions_tags[-1][:1] == 'O':
            batch_predictions[j, [id for id, tag in id2tag.items() if tag[:1] not in ['B', 'O']]] = -100
        # if last prediction was B-XXX, then only accepts I-XXX, O or B-XYZ
        elif batch_predictions_tags[-1][:1] == 'B':
            batch_predictions[j, [id for id, tag in id2tag.items() if tag not in [f'I-{batch_predictions_tags[-1][2:]}',
                                                                                  'O'] and tag[:1] != 'B']] = -100
        # if last prediction was I-XXX, then only accepts I-XXX, B or O
        elif batch_predictions_tags[-1][:1] == 'I':
            batch_predictions[j, [id for id, tag in id2tag.items() if tag not in [f'I-{batch_predictions_tags[-1][2:]}']
                                  and tag[:1] not in ['B', 'O']]] = -100

        return id2tag[np.argmax(batch_predictions[j])]

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        preds_masked = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i, j]])
                    preds_list[i].append(id2tag[np.argmax(predictions[i][j])])
                    preds_masked[i].append(mask_output_iob2(predictions[i], j, preds_masked[i]))

        return preds_list, preds_masked, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, preds_masked, out_label_list = align_predictions(p.predictions, p.label_ids)
        print(performance_measure(out_label_list,
                                  preds_list))  # se colocar no retorno, fica com warning na escrita para o tensorboard - Verificar no huggingface
        print(classification_report(out_label_list,
                                    preds_list))  # se colocar no retorno, fica com warning na escrita para o tensorboard - Verificar no huggingface
        print(
            metrics.flat_classification_report(y_true=out_label_list, y_pred=preds_list, labels=sorted(tag2id.keys())))

        with open('data/eval_preds_pub.pickle', 'wb') as f:
            pickle.dump(preds_list, f)

        with open('data/eval_preds_masked_pub.pickle', 'wb') as f:
            pickle.dump(preds_masked, f)

        with open('data/eval_labels_pub.pickle', 'wb') as f:
            pickle.dump(out_label_list, f)

        print("f1", f1_score(out_label_list, preds_list), "f1_masked", f1_score(out_label_list, preds_masked))

        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "f1_masked": f1_score(out_label_list, preds_masked)
        }

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset,
        compute_metrics=compute_metrics
    )

    print(trainer.train())
    print(trainer.evaluate())

    trainer.save_model(f'models/{experiment_name}')
    print()


def save_data(dataset: List[List[int]], filename: str):
    with open(f'data/{filename}.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def load_data(filenames: List[str]):
    contents = []
    for filename in filenames:
        with open(f'data/{filename}.pickle', 'rb') as f:
            contents.append(pickle.load(f))
    return contents


def get_data(tokenizer):
    if Path('data/train_texts_pub.pickle').exists() and Path('data/train_tags_pub.pickle').exists() \
            and Path('data/val_texts_pub.pickle').exists() and Path('data/val_tags_pub.pickle').exists():
        train_texts, val_texts, train_tags, val_tags = load_data(['train_texts_pub', 'val_texts_pub', 'train_tags_pub', 'val_tags_pub'])
    else:
        textos, tags = ProcessDataset('labeled_4_labels_pub.jsonl', False).get_dataset()
        assert len(textos) == len(tags)
        for texto, tag in zip(textos, tags):
            assert len(texto) == len(tag)

        textos, tags = ProcessDataset.pre_processar_base(textos, tags, tokenizer)
        train_texts, val_texts, train_tags, val_tags = train_test_split(textos, tags, test_size=.2, random_state=42)
        save_data(train_texts, 'train_texts_pub_multi')
        save_data(train_tags, 'train_tags_pub_multi')
        save_data(val_texts, 'val_texts_pub_multi')
        save_data(val_tags, 'val_tags_pub_multi')

    return train_texts, val_texts, train_tags, val_tags


if __name__ == '__main__':
    run()
