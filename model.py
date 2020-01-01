from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertOnlyMLMHead
from torch.nn import CrossEntropyLoss, Linear, LogSoftmax, ReLU
from util import get_candidate_vocab_mask
import torch

TARGET_VOCAB_SIZE = 1934

class MaskedModel(BertPreTrainedModel):
    def __init__(self, config):        
        super(MaskedModel, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)        
        # self.softmax = LogSoftmax(dim=1)
        self.cls2 = Linear(config.vocab_size, TARGET_VOCAB_SIZE)        
        self.relu = ReLU()
        self.init_weights()
        # self.vocab_mask = get_candidate_vocab_mask('./dataset/', config.vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # if self.vocab_mask is not None:             
        #     prediction_scores.view(-1, self.config.vocab_size)[:, ~self.vocab_mask] = -1e5
        prediction_scores = self.relu(prediction_scores)
        prediction_scores = self.cls2(prediction_scores)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)