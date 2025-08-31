# THis part building the clip encoder, exntract the representation from image and text
# model link: https://huggingface.co/openai/clip-vit-large-patch14
## https://arxiv.org/abs/2103.00020
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.get('device','cuda')
        model_name = config.get('encoder_model_name', "openai/clip-vit-base-patch32")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)

        # Freeze the params
        for param in self.clip.parameters():
            param.requires_grad = False


    def forward(self, image=None, caption=None):

        outputs = {}

        if image != None:

            if not isinstance(image, list):
                image = [image]

            image_inputs = self.processor(images=image,return_tensors='pt')['pixel_values'].to(self.device)
            vision_encoder = self.clip.vision_model(pixel_values = image_inputs)
            pooled_output = vision_encoder.pooler_output
            # L2 Normlise
            outputs['image_feat'] = pooled_output
        if caption != None:
            text_inputs = self.processor(
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_outputs = self.clip.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            caption_cls = text_outputs.pooler_output

            outputs['caption_feat'] = caption_cls

        return outputs['image_feat'], outputs['caption_feat']
