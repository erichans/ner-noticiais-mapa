from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Tuple, List
import numpy as np

import torch
from torch import nn
from torchcrf import CRF


class BertCRF(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # loss_weight = [1.] * self.num_labels
        # loss_weight[config.label2id['O']] = 0.01
        # loss_weight[config.label2id['B-ORG']] = 1000
        # loss_weight[config.label2id['I-ORG']] = 1000
        # loss_weight[config.label2id['E-ORG']] = 1000

        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

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
        # sequence_output = self.dropout(sequence_output)

        # first_token_tensor = sequence_output[:, 0]
        # pooled_output = self.pooler(first_token_tensor)
        # pooled_output = self.pooler_activation(pooled_output)
        # logits = self.classifier(pooled_output)

        logits = self.classifier(sequence_output)

        # Only keep active parts of the loss
        full_mask = (attention_mask == 1) & (labels != -100)

        loss = 0
        for logit, label, mask in zip(logits, labels, full_mask):
            loss += -self.crf(logit[mask].unsqueeze(0), label[mask].unsqueeze(0), reduction='token_mean')

        loss /= logits.shape[0]
        # loss = self.loss_function(self.align_predictions(active_logits, active_labels))
        # loss = self.loss_function(active_logits, active_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

    def mask_output(self, batch_predictions: np.ndarray, j: int, batch_predictions_tags: List):
        # First token: possible predictions: O, B-*. Others = -100 or
        # if last prediction was E-XXX, then only accepts B-XYZ or O or
        # if last prediction was O, then only accepts B-XYZ or O
        if len(batch_predictions_tags) == 0 or self.config.id2label[batch_predictions_tags[-1]][:1] in ['E', 'O']:
            batch_predictions[j, [id for id, tag in self.config.id2label.items() if tag[:1] not in ['B', 'O']]] = -100
        # if last prediction was B-XXX, then only accepts I-XXX, E-XXX, O or B-XYZ
        elif self.config.id2label[batch_predictions_tags[-1]][:1] == 'B':
            batch_predictions[j, [id for id, tag in self.config.id2label.items()
                                  if tag not in [f'I-{self.config.id2label[batch_predictions_tags[-1]][2:]}',
                                                 f'E-{self.config.id2label[batch_predictions_tags[-1]][2:]}',
                                                 'O'] and tag[:1] != 'B']] = -100
        # if last prediction was I-XXX, then only accepts I-XXX, E-XXX
        elif self.config.id2label[batch_predictions_tags[-1]][:1] == 'I':
            batch_predictions[j, [id for id, tag in self.config.id2label.items()
                                  if tag not in [f'I-{self.config.id2label[batch_predictions_tags[-1]][2:]}',
                                                 f'E-{self.config.id2label[batch_predictions_tags[-1]][2:]}']]] = -100

        return np.argmax(batch_predictions[j])

    def align_predictions(self, predictions, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = torch.argmax(predictions, dim=-1)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_ids[i, j])
                    preds_list[i].append(self.mask_output(predictions[i], j, preds_list[i]))

        return preds_list, out_label_list
