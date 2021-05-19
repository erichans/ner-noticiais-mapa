from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Tuple, List
import numpy as np

import torch
from torch import nn


class BertNER(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        loss_weight = [1.] * self.num_labels
        # loss_weight[config.label2id['O']] = 0.01
        # loss_weight[config.label2id['B-ORG']] = 10
        # loss_weight[config.label2id['B-LOC']] = 10
        # loss_weight[config.label2id['B-PESSOA']] = 10
        # loss_weight[config.label2id['B-PUB']] = 10
        self.loss_function = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        # sequence_output = outputs[0]
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        # logits, labels = self.align_predictions(logits, labels)

        loss = 0
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is None:
                loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_function.ignore_index).type_as(labels))

                loss = self.loss_function(active_logits, active_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def mask_output(self, batch_predictions: np.ndarray, j: int, batch_predictions_tags: List[torch.tensor],
                    label_ids: torch.tensor):
        # First token: possible predictions: O, B-*. Others = -100 or
        # if last prediction was E-XXX, then only accepts B-XYZ or O or
        # if last prediction was O, then only accepts B-XYZ or O
        if len(batch_predictions_tags) == 0 or self.config.id2label[batch_predictions_tags[-1].item()][:1] in ['E', 'O']:
            forbidden_indices = [id for id, tag in self.config.id2label.items() if tag[:1] not in ['B', 'O']]
            batch_predictions[j, forbidden_indices] = 0
        # if last prediction was B-XXX, then only accepts I-XXX, E-XXX, O or B-XYZ
        elif self.config.id2label[batch_predictions_tags[-1].item()][:1] == 'B':
            forbidden_indices = [id for id, tag in self.config.id2label.items()
                                 if tag not in [f'I-{self.config.id2label[batch_predictions_tags[-1].item()][2:]}',
                                                f'E-{self.config.id2label[batch_predictions_tags[-1].item()][2:]}', 'O']
                                 and tag[:1] != 'B']
            batch_predictions[j, forbidden_indices] = 0
        # if last prediction was I-XXX, then only accepts I-XXX, E-XXX
        elif self.config.id2label[batch_predictions_tags[-1].item()][:1] == 'I':
            forbidden_indices = [id for id, tag in self.config.id2label.items()
                                 if tag not in [f'I-{self.config.id2label[batch_predictions_tags[-1].item()][2:]}',
                                                f'E-{self.config.id2label[batch_predictions_tags[-1].item()][2:]}']]
            batch_predictions[j, forbidden_indices] = 0

        return batch_predictions[j]

    def align_predictions(self, predictions: torch.tensor, label_ids: torch.tensor) -> Tuple[List, List]:
        preds = torch.argmax(predictions, dim=-1)
        batch_size, seq_len = preds.shape
        # out_label_list = [[] for _ in range(batch_size)]
        out_label_list = torch.zeros(size=(batch_size, seq_len), device=predictions.device).long()
        # preds_list = [[] for _ in range(batch_size)]
        preds_list = torch.zeros(size=(batch_size, seq_len), device=predictions.device).float()

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i][j] = label_ids[i, j]
                    predictions[i][j] = self.mask_output(predictions[i], j, preds_list[i], label_ids[i])
                    preds_list[i] = torch.argmax(predictions[i][j])

        return predictions, out_label_list
