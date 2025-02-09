import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, ViTModel

# img transformer
class Transformer_One(nn.Module):
    def __init__(self, vit_model_name, embed_dim, device="cuda", freeze_vit=False):
        super(Transformer_One, self).__init__()
        self.device = device
        self.model = ViTModel.from_pretrained(vit_model_name).to(self.device) # 'google/vit-base-patch16-224'

        hidden_size = self.model.config.hidden_size

        if freeze_vit:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True # default

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.model(images)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embeddings)  # size=(batch_size, embed_dim)
        return embeddings


# txt transformer
class Transformer_Two(nn.Module):
    def __init__(self, bert_model_name, embed_dim, device="cuda", freeze_bert=False):
        super(Transformer_Two, self).__init__()
        self.device = device 
        self.tokenizer =  BertTokenizer.from_pretrained(bert_model_name) # 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(bert_model_name).to(self.device)

        hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True # default 

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_text):
        encoded_inputs = self.tokenizer(
            input_text, padding=True,
            truncation=True,
            return_tensors="pt").to(self.device)

        outputs = self.bert(**encoded_inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :] # cls
        embeddings = self.projection(cls_embeddings)  
        return embeddings
