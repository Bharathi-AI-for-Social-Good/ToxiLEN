import torch
import torch.nn as nn
from models.encoders.clip_encoder import CLIPEncoder
from models.encoders.text_encoder import TokenModel, RobertaEncoder
from models.layers.cross_attn import CrossAttentionBlock
from models.layers.bi_gate import BiGateFusion
from models.layers.prefix_uniform import PrefixUniformLayer
from models.layers.focal_loss import FocalLoss
from torch.nn import functional as F
from app.utils.logger import get_logger

logger = get_logger()

class MemeMultimodalDetector(nn.Module):
    def __init__(self, config, variant):
        super().__init__()

        self.variant = variant

        self.clip_encoder = CLIPEncoder(config)
        self.prompt_encoder = TokenModel(config)
        self.explain_encoder = RobertaEncoder(config)


        self.projection_layer = PrefixUniformLayer(config, variant)

        self.cross_atten = CrossAttentionBlock(dim=768)

        self.bi_gate = BiGateFusion(hidden_dim=1024)

        self.s_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.classifier = nn.Linear(1024, 2)

        self.f_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    def forward(self, image, prompt, explanation, caption, labels=None):
        # image_feat (B, 1024), caption_feat(B,768)
        img_feat, caption_feat = self.clip_encoder(image=image, caption=caption)


        prompt_feat = self.prompt_encoder(prompt, return_tokens=True)
        explanation_feat = self.explain_encoder(explanation, return_tokens=True)


        I,C,P,E = self.projection_layer(img_feat, caption_feat, prompt_feat, explanation_feat)

        # clip_feat(B, 1024)
        clip_feat = I * C
        # cross_embedding(B,1024)
        cross_embedding = self.cross_atten(P, E, E)
        cross_embedding = cross_embedding.mean(dim=1)  # [B, 1024]

        # fused (B,2048)
        fused = self.bi_gate(clip_feat, cross_embedding)

        F_output = self.s_mlp(fused)                # (B, 1024)
        logits = self.classifier(F_output)          # (B, 2)
        log_probs = F.log_softmax(logits, dim=-1)

        if labels is not None:
          loss = self.f_loss_fn(log_probs, labels)
          return log_probs, loss
        else:
          return log_probs

        
        
        
        
        
