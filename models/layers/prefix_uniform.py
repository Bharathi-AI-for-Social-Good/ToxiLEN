import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixUniformLayer(nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        proj_dim = config['proj_dim']

        self.project_clip_img = nn.Sequential(
            nn.Linear(config['clip_img_dim'], 1024),
            nn.LayerNorm(proj_dim)
        )

        self.project_caption = nn.Sequential(
            nn.Linear(config['clip_text_dim'], 1024),
            nn.LayerNorm(proj_dim)
        )

        self.project_prompt = nn.Sequential(
            nn.Linear(config['caption_dim'], 1024),
            nn.LayerNorm(proj_dim)
        )
        self.project_explanation = nn.Sequential(
            nn.Linear(config['caption_dim'], 1024),
            nn.LayerNorm(proj_dim)
        )
        self.variant = variant

    def forward(self, img, caption,prompt, explanation):
        I = self.project_clip_img(img)
        C = self.project_caption(caption)
        P = self.project_prompt(prompt)
        E = self.project_explanation(explanation)

        return I, C, P, E