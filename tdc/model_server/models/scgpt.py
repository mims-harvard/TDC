from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Dict


class ScGPTConfig(PretrainedConfig):
    model_type = "scgpt"

    def __init__(
            self,
            vocab_size=60697,
            embsize=512,
            d_hid=512,
            nlayers=12,
            nhead=8,
            max_seq_len=1536,
            dropout=0.0,
            pad_token_id=0,
            use_fast_transformer=True,
            input_emb_style="continuous",
            cell_emb_style="cls",  # output embedding vector with 
            norm_scheme="post",
            explicit_zero_prob=False,
            use_flash_attention=True,
            **kwargs):
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.use_fast_transformer = use_fast_transformer
        if input_emb_style not in ["continuous"]:
            raise ValueError(
                f"Invalid input_emb_style: {input_emb_style}. Only continuous embeddings currently supported."
            )
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.norm_scheme = norm_scheme
        self.explicit_zero_prob = explicit_zero_prob
        self.use_flash_attention = self.use_fast_transformer and torch.cuda.is_available(
        ) and use_flash_attention
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class ExprDecoder(nn.Module):

    def __init__(self, d_model: int, explicit_zero_prob: bool = False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),  # we don't use batch labels
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_value = self.fc(x).squeeze(-1)
        if not self.explicit_zero_prob:
            return {"pred": pred_value}
        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return {
            "pred": pred_value,
            "zero_probs": zero_probs
        }  # TODO: what about inference / bernoulli?


class FlashTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_scheme="post"):
        super().__init__()
        from flash_attn.flash_attention import FlashMHA

        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                          nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_scheme = norm_scheme


# Helper class to ensure we have the correct attention structure
class MultiheadAttentionWithBias(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Combined input projections for Q, K, V
        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters following PyTorch's MultiheadAttention initialization
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        return nn.functional.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            None,
            None,
            None,  # No bias_k, bias_v, or add_zero_attn
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            batch_first=self.batch_first)[0]


from transformers import PreTrainedModel
import torch.nn as nn


class ScGPTPreTrainedModel(PreTrainedModel):
    config_class = ScGPTConfig
    base_model_prefix = "scgpt"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ScGPTModel(ScGPTPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        # Gene name embeddings remain the same
        self.gene_encoder = nn.ModuleDict({
            "embedding":
                nn.Embedding(config.vocab_size,
                             config.embsize,
                             padding_idx=config.pad_token_id),
            "enc_norm":
                nn.LayerNorm(config.embsize)
        })

        # Value encoder remains the same
        if config.input_emb_style == "continuous":
            self.value_encoder = nn.ModuleDict({
                "linear1": nn.Linear(1, config.embsize),
                "linear2": nn.Linear(config.embsize, config.embsize),
                "norm": nn.LayerNorm(config.embsize),
                "dropout": nn.Dropout(config.dropout)
            })
        elif config.input_emb_style == "scaling":
            self.value_encoder = nn.Identity()
            raise Exception(
                "scaling input embedding style not supported because this model was trained on continuous style"
            )
        else:
            raise Exception("unsupported embedding style")

        # Modified transformer layers to use combined QKV projections
        # self.transformer = nn.ModuleDict({
        #     "layers": nn.ModuleList([
        #         nn.ModuleDict({
        #             "self_attn": MultiheadAttentionWithBias(
        #                 config.embsize,
        #                 config.nhead,
        #                 dropout=config.dropout,
        #                 batch_first=True
        #             ),
        #             "linear1": nn.Linear(config.embsize, config.d_hid),
        #             "linear2": nn.Linear(config.d_hid, config.embsize),
        #             "norm1": nn.LayerNorm(config.embsize),
        #             "norm2": nn.LayerNorm(config.embsize),
        #         }) for _ in range(config.nlayers)
        #     ])
        # })

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=config.embsize,
                nhead=config.nhead,
                dim_feedforward=config.d_hid,
                dropout=config.dropout,
                batch_first=True,  # just for replication
            ),
            num_layers=config.nlayers)

        # Decoder remains the same
        self.expr_decoder = ExprDecoder(config.embsize,
                                        config.explicit_zero_prob)

        # we ignore cls_decoder because we do not pursue classification task
        # we also ignore mvc and similarity because we ignore generative tasks

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_cell_emb: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Tensor of gene indices, shape [batch_size, seq_len]
            values: Tensor of expression values, shape [batch_size, seq_len]
            attention_mask: Optional mask tensor, shape [batch_size, seq_len]
            output_cell_emb: Whether to output cell embeddings

        Returns:
            Dictionary containing:
                - 'pred': Predicted expression values
                - 'cell_emb': Cell embeddings (if output_cell_emb=True)
                - 'zero_probs': Zero probabilities (if config.explicit_zero_prob=True)
        """
        # Gene embeddings
        gene_emb = self.gene_encoder["embedding"](input_ids)
        gene_emb = self.gene_encoder["enc_norm"](gene_emb)

        # Value encoding
        if hasattr(self, 'value_encoder'):
            values = values.unsqueeze(-1)  # Add feature dimension
            value_emb = self.value_encoder["linear1"](values)
            if "activation" in self.value_encoder:
                value_emb = self.value_encoder["activation"](value_emb)
            value_emb = self.value_encoder["linear2"](value_emb)
            value_emb = self.value_encoder["norm"](value_emb)
            value_emb = self.value_encoder["dropout"](value_emb)

            if self.config.input_emb_style == "continuous":
                hidden_states = gene_emb + value_emb
            else:  # "scaling", currrently not supported
                hidden_states = gene_emb * value_emb
        else:
            hidden_states = gene_emb

        # Convert attention_mask for transformer
        # Flash attention expects mask of 0s for tokens to attend to and 1s for tokens to ignore
        # if self.use_flash_attention and attention_mask is not None:
        #     if attention_mask.dtype != torch.bool:
        #         attention_mask = attention_mask.bool()
        #     attention_mask = ~attention_mask # we assume user follows huggingface convention for the attention mask

        # # Apply transformer layers
        # if self.use_flash_attention:
        #     for layer in self.transformer:
        #         hidden_states = layer(
        #             hidden_states,
        #             src_key_padding_mask=attention_mask
        #         )
        # else:
        hidden_states = self.transformer(hidden_states,
                                         src_key_padding_mask=attention_mask)

        # Get cell embeddings if requested
        output_dict = {}
        if output_cell_emb:
            if self.config.cell_emb_style == "cls":
                cell_emb = hidden_states[:, 0]
            elif self.config.cell_emb_style == "avg-pool":
                cell_emb = hidden_states.mean(dim=1)
            else:  # w-pool
                # Weighted pooling using input values as weights
                weights = F.softmax(values, dim=1).unsqueeze(-1)
                cell_emb = (hidden_states * weights).sum(dim=1)
            output_dict['cell_emb'] = cell_emb

        # Decode expression values
        decoder_output = self.expr_decoder(hidden_states)
        output_dict.update(decoder_output)

        return output_dict

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            # Additional initialization for linear layers
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
