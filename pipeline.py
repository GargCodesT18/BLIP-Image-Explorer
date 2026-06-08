
import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Caption Model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    cache_dir="./model"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    cache_dir="./model"
).to(device)

# VQA Model
vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base",
    cache_dir="./model"
)

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    cache_dir="./model"
).to(device)

pipeline = {
    "device": device,
    "processor": processor,
    "caption_model": caption_model,
    "vqa_processor": vqa_processor,
    "vqa_model": vqa_model
}
