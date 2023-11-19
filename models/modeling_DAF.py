import os
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel, BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertModel, BertPredictionHeadTransform
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput, \
    QuestionAnsweringModelOutput, TokenClassifierOutput
        
from models.pinyin_embedding import PinyinEmbedding
from models.modeling_glycebert import GlyceBertModel
from datasets.utils import Pinyin

SMALL_CONST = 1e-15

class Pinyin_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pinyin = Pinyin()
        self.transform = BertPredictionHeadTransform(config)
        self.initial_classifier = nn.Linear(config.hidden_size, self.pinyin.sm_size)
        self.final_classifier = nn.Linear(config.hidden_size, self.pinyin.ym_size)
        self.tone_classifier = nn.Linear(config.hidden_size, self.pinyin.sd_size)

    def forward(self, pinyin_output):
        pinyin_output = self.transform(pinyin_output)
        initial_scores = self.initial_classifier(pinyin_output)
        final_scores = self.final_classifier(pinyin_output)
        tone_scores = self.tone_classifier(pinyin_output)
        return initial_scores, final_scores, tone_scores


class PredictHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.Pinyin_relationship = Pinyin_Classifier(config)
    
    def forward(self, sequence_output, pinyin_output):
        sequence_scores = self.predictions(sequence_output)
        initial_scores, final_scores, tone_scores = self.Pinyin_relationship(pinyin_output)
        return sequence_scores, initial_scores, final_scores, tone_scores


class Vanilla_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_a,
        hidden_states_b,
        attention_mask=None,
    ):
        
        query_layer = self.query(hidden_states_a)
        key_layer = self.key(hidden_states_b)
        value_layer = self.value(hidden_states_b)

        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        query_layer = self.transpose_for_scores(query_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 
        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 1)
            attention_mask = attention_mask.expand(-1, attention_scores.size(1), -1)
            attention_mask = torch.unsqueeze(attention_mask, 3)
            attention_mask = attention_mask.expand(-1, -1, -1, attention_scores.size(3))
            
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer

        return outputs


class Differential_Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states_a, hidden_states_b, attention_mask=None):
        
        query_layer = self.query(hidden_states_a)
        _, seq_len, _ = query_layer.shape
        key_layer = self.key(hidden_states_b)
        value_layer = self.value(hidden_states_b)
        
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        query_layer = self.transpose_for_scores(query_layer)
        
        attention_scores = \
        torch.sum(
            torch.unsqueeze(query_layer, 3) - 
            torch.unsqueeze(key_layer, 3).expand(-1, -1, -1, seq_len, -1), dim=-1)
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 
        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 1)
            attention_mask = attention_mask.expand(-1, attention_scores.size(1), -1)
            attention_mask = torch.unsqueeze(attention_mask, 3)
            attention_mask = attention_mask.expand(-1, -1, -1, attention_scores.size(3))
            
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = context_layer

        return outputs


class Coupled_Attention_Module(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.Tanh = nn.Tanh()
        self.SoftMax = nn.Softmax(dim=1)
        self.van_att = Vanilla_Attention(config)
        self.diff_att = Differential_Attention(config)

        self.van_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.WD_fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False) 

        self.d_theta_fc = nn.Linear(config.hidden_size * 2, config.hidden_size) 

        self.WV_fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.diff_fc = nn.Linear(config.hidden_size, config.hidden_size) 

        self.v_gamma_fc = nn.Linear(config.hidden_size *2, config.hidden_size) 
        
        self.diff_out_fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.van_out_fc = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.diff_fusion_fc = nn.Linear(config.hidden_size, config.hidden_size) 
        self.van_fusion_fc = nn.Linear(config.hidden_size, config.hidden_size) 

        self.fusion_gate = nn.Linear(config.hidden_size * 2, 1, bias=False) 

        self.noise_filter_fc = nn.Linear(config.hidden_size, config.hidden_size) 
        self.noise_filter_out_fc = nn.Linear(config.hidden_size * 2, 1, bias=False) 

        self.final_out_fc = nn.Linear(config.hidden_size, config.hidden_size) 

    def forward(self, hidden_states_x, hidden_states_y, attention_mask=None):
        
        van_vector = self.van_att(hidden_states_x, hidden_states_y, attention_mask)
        diff_vector = self.diff_att(hidden_states_x, hidden_states_y, attention_mask)
        
        assert van_vector.shape == diff_vector.shape
        theta = self.Tanh(torch.cat((self.WD_fc(diff_vector), self.van_fc(van_vector)), dim=-1))
        
        d_theta = diff_vector * self.SoftMax(self.d_theta_fc(theta)) 
        gamma = self.Tanh(torch.cat((self.WV_fc(van_vector), self.diff_fc(d_theta)), dim=-1))
        a_gamma = van_vector * self.SoftMax(self.v_gamma_fc(gamma))
        
        diff_out = self.Tanh(self.diff_out_fc(torch.cat((diff_vector, d_theta), dim=-1)))
        van_out = self.Tanh(self.van_out_fc(torch.cat((van_vector, a_gamma), dim=-1)))
        
        diff_fusion = self.Tanh(self.diff_fusion_fc(diff_out))
        van_fusion = self.Tanh(self.van_fusion_fc(van_out))
        
        fusion_gate = torch.sigmoid(self.fusion_gate(torch.cat((diff_fusion, van_fusion), dim=-1)))
        
        fusion_feature = fusion_gate * van_fusion + (1-fusion_gate) * diff_fusion
        
        noise_filter = torch.sigmoid(self.noise_filter_out_fc(torch.cat((van_vector, self.noise_filter_fc(fusion_feature)), dim=-1)))

        final_out = noise_filter * self.Tanh(self.final_out_fc(fusion_feature))

        return final_out


class Adaptive_Fusion_Module(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.filter_out_fc = nn.Linear(config.hidden_size * 2, 1, bias=False)

    def forward(self, hidden_states_a, hidden_states_b):

        gate = torch.sigmoid(self.filter_out_fc(torch.cat((hidden_states_a, hidden_states_b), dim=-1)))
        return hidden_states_a * gate + hidden_states_b * (1 - gate) 


class DAF_model(BertPreTrainedModel):
    def __init__(self, config):
        super(DAF_model, self).__init__(config)

        self.word_encoder = GlyceBertModel(config)
        self.pinyin_encoder = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size,
                                                 config_path=os.path.join(config.name_or_path, 'config'))
        self.cls = PredictHeads(config)

        self.cou_att_pinyin_word = Coupled_Attention_Module(config) 
        self.cou_att_word_pinyin = Coupled_Attention_Module(config) 

        self.pinyin_word_filter = Adaptive_Fusion_Module(config)
        self.word_pinyin_filter = Adaptive_Fusion_Module(config)

        self.loss_fct = CrossEntropyLoss(reduction='none')

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        tgt_pinyin_ids=None,
        pinyin_labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0)*(input_ids != 101)*(input_ids != 102).long()
        word_outputs_x = self.word_encoder(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        word_encoded_x = word_outputs_x[0]
        pinyin_encoded_x = self.pinyin_encoder(pinyin_ids) 

        pinyin_word_out_x = self.cou_att_pinyin_word(pinyin_encoded_x, word_encoded_x, attention_mask=attention_mask)
        word_pinyin_out_x = self.cou_att_word_pinyin(word_encoded_x, pinyin_encoded_x, attention_mask=attention_mask)
        
        sequence_out = self.pinyin_word_filter(pinyin_word_out_x, word_encoded_x)

        pinyin_out_x = self.word_pinyin_filter(word_pinyin_out_x, pinyin_encoded_x)

        if tgt_pinyin_ids is not None:
            with torch.no_grad():
                word_outputs_y = self.word_encoder(
                    labels,
                    tgt_pinyin_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                word_encoded_y = word_outputs_y[0]
                pinyin_encoded_y = self.pinyin_encoder(tgt_pinyin_ids)
                
                word_pinyin_out_y = self.cou_att_word_pinyin(word_encoded_y, pinyin_encoded_y, attention_mask=attention_mask)

                pinyin_out_y = self.word_pinyin_filter(word_pinyin_out_y, pinyin_encoded_y)

                pron_x = self.cls.Pinyin_relationship.transform(pinyin_out_x)
                pron_y = self.cls.Pinyin_relationship.transform(pinyin_out_y)    
    
                sim_xy = F.cosine_similarity(pron_x, pron_y, dim=-1)    
                factor = torch.exp(-((sim_xy - 1.0)).pow(2)).detach()

        sequence_scores, initial_scores, final_scores, tone_scores = self.cls(sequence_out, pinyin_out_x)
        
        text_loss = None
        loss_fct = self.loss_fct
        if labels is not None and pinyin_labels is not None:

            active_loss = loss_mask.view(-1) == 1

            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            text_loss = loss_fct(sequence_scores.view(-1, self.config.vocab_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            initial_loss = loss_fct(initial_scores.view(-1, self.cls.Pinyin_relationship.pinyin.initial_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            final_loss = loss_fct(final_scores.view(-1, self.cls.Pinyin_relationship.pinyin.final_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            tone_loss = loss_fct(tone_scores.view(-1, self.cls.Pinyin_relationship.pinyin.tone_size), active_labels)

            def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)

            text_loss = weighted_mean(torch.ones_like(text_loss), text_loss)
            pinyin_loss = weighted_mean(factor.view(-1), (initial_loss + final_loss + tone_loss)/3)

        loss = text_loss + pinyin_loss

        return MaskedLMOutput(
            loss=loss,
            logits=sequence_scores,
            hidden_states=word_encoded_x.hidden_states,
            attentions=word_encoded_x.attentions,
        )
