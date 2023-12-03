import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import ElectraTokenizer, ElectraModel, LayoutLMModel, LayoutLMTokenizer, ReformerModelWithLMHead, ReformerTokenizer, CamembertModel, CamembertTokenizer
import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2


def clip32(dataframe, images_path):
    
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    combined_features = []

    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        
        image = Image.open(path).convert("RGB")
        image_input = clip_processor(images=image, return_tensors="pt")
        
        text = dataframe.tweet_text.iloc[i]
        text_input = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_input)
            text_features = clip_model.get_text_features(**text_input)
        
        image_features_reshaped = image_features.view(image_features.shape[0], -1).numpy()
        text_features_pooled = text_features.view(text_features.shape[0], -1).numpy()
        combined_features.append(np.concatenate((text_features_pooled, image_features_reshaped), axis=-1))
    
    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MaxAbsScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    return combined_features_scaled

def clip14(dataframe, images_path):
    
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    combined_features = []

    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        
        image = Image.open(path).convert("RGB")
        image_input = clip_processor(images=image, return_tensors="pt")
        
        text = dataframe.tweet_text.iloc[i]
        text_input = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_input)
            text_features = clip_model.get_text_features(**text_input)
        
        image_features_reshaped = image_features.view(image_features.shape[0], -1).numpy()
        text_features_pooled = text_features.view(text_features.shape[0], -1).numpy()
        combined_features.append(np.concatenate((text_features_pooled, image_features_reshaped), axis=-1))
    
    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MaxAbsScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    return combined_features_scaled

def convnext_small_rel(dataframe, images_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    convnext_model = models.convnext_small(pretrained=True)
    convnext_model = convnext_model.to(device)
    convnext_model.eval()

    reformer_model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
    reformer_model = reformer_model.to(device)
    reformer_tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
    if reformer_tokenizer.pad_token is None:
        reformer_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    reformer_model.eval()

    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')
    electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = electra_model.to(device)
    electra_model.eval()    
    
    layoutlm_model = LayoutLMModel.from_pretrained('microsoft/layoutlm-large-uncased')
    layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-large-uncased')
    layoutlm_model = layoutlm_model.to(device)
    layoutlm_model.eval()
    
    combined_features = []

    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        img = Image.open(path).convert("RGB")
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        image_features = convnext_model(img)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = image_features.detach().cpu().numpy()

        text = dataframe.tweet_text.iloc[i]
        reformer_text_input = reformer_tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding='longest', return_tensors='pt', return_attention_mask=True)
        reformer_input_ids = reformer_text_input['input_ids'].to(device)
        reformer_attention_mask = reformer_text_input['attention_mask'].to(device)

        with torch.no_grad():
            reformer_text_features = reformer_model(input_ids=reformer_input_ids, attention_mask=reformer_attention_mask)[0][:, 0, :]
        reformer_text_features = reformer_text_features.detach().cpu().numpy()
        
        electra_text_tokens = electra_tokenizer.encode(text, add_special_tokens=True)
        electra_text_input = torch.tensor(electra_text_tokens).unsqueeze(0).to(device)
        electra_text_features = electra_model(electra_text_input)[0]
        electra_text_features = torch.mean(electra_text_features, dim=1).detach().cpu().numpy()
        
        layoutlm_inputs = layoutlm_tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        layoutlm_input_ids = layoutlm_inputs['input_ids'].to(device)
        layoutlm_attention_mask = layoutlm_inputs['attention_mask'].to(device)
        layoutlm_text_features = layoutlm_model(layoutlm_input_ids, attention_mask=layoutlm_attention_mask)[0]
        layoutlm_text_features = torch.mean(layoutlm_text_features, dim=1).detach().cpu().numpy()
        
        text_features = np.concatenate((electra_text_features, reformer_text_features, layoutlm_text_features), axis=-1)
        combined_features.append(np.concatenate((text_features, image_features), axis=-1))

    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MaxAbsScaler()
    combined_features_transformed = scaler.fit_transform(combined_features)

    return combined_features_transformed

def swin_v2_s_camembert_base(dataframe, images_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    swin_model = models.swin_v2_s(pretrained=True)
    swin_model = swin_model.to(device)
    swin_model.eval()

    camembert_model = CamembertModel.from_pretrained("camembert/camembert-base")
    camembert_model = camembert_model.to(device)
    camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
    camembert_model.eval()
    
    combined_features = []

    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        img = Image.open(path).convert("RGB")
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        
        swin_image_features = swin_model(img)
        swin_image_features = swin_image_features.view(swin_image_features.size(0), -1)
        image_features = swin_image_features.detach().cpu().numpy()
        
        text = dataframe.tweet_text.iloc[i]

        camembert_text_input = camembert_tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding='longest', return_tensors='pt')
        camembert_input_ids = camembert_text_input['input_ids'].to(device)
        camembert_attention_mask = camembert_text_input['attention_mask'].to(device)

        with torch.no_grad():
            camembert_text_features = camembert_model(input_ids=camembert_input_ids, attention_mask=camembert_attention_mask)[0][:, 0, :]
        text_features = camembert_text_features.detach().cpu().numpy()
        
        combined_features.append(np.concatenate((text_features, image_features), axis=-1))

    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MaxAbsScaler()
    combined_features_transformed = scaler.fit_transform(combined_features)

    return combined_features_transformed

def swin_v2_s_camembert_base_OD(dataframe, images_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    swin_model = models.swin_v2_s(pretrained=True)
    swin_model = swin_model.to(device)
    swin_model.eval()

    camembert_model = CamembertModel.from_pretrained("camembert/camembert-base")
    camembert_model = camembert_model.to(device)
    camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
    camembert_model.eval()

    scene_model = fasterrcnn_mobilenet_v3_large_fpn(pretrained = True)
    scene_model = scene_model.to(device)
    scene_model.eval()

    combined_features = []
    scene_feature_size = 100
    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        img = Image.open(path).convert("RGB")
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        swin_image_features = swin_model(img)
        swin_image_features = swin_image_features.view(swin_image_features.size(0), -1)
        image_features = swin_image_features.detach().cpu().numpy()

        text = dataframe.tweet_text.iloc[i]
        camembert_text_input = camembert_tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding='longest', return_tensors='pt')
        camembert_input_ids = camembert_text_input['input_ids'].to(device)
        camembert_attention_mask = camembert_text_input['attention_mask'].to(device)

        with torch.no_grad():
            camembert_text_features = camembert_model(input_ids=camembert_input_ids, attention_mask=camembert_attention_mask)[0][:, 0, :]
        text_features = camembert_text_features.detach().cpu().numpy()
        
        with torch.no_grad():
            scene_outputs = scene_model(img)
        scene_features = scene_outputs[0]['boxes'].view(-1).detach().cpu().numpy()
        scene_features = scene_features[:scene_feature_size]
        while len(scene_features) < scene_feature_size:
            scene_features = np.append(scene_features, 0)
        scene_features = scene_features.reshape(1, -1)

        combined_features.append(np.concatenate((text_features, image_features, scene_features), axis=-1))

    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MaxAbsScaler()
    combined_features_transformed = scaler.fit_transform(combined_features)

    return combined_features_transformed

def convnext_small_re_OD(dataframe, images_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    convnext_model = models.convnext_small(pretrained=True)
    convnext_model = convnext_model.to(device)
    convnext_model.eval()
    
    reformer_model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
    reformer_model = reformer_model.to(device)
    reformer_tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")
    
    if reformer_tokenizer.pad_token is None:
        reformer_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    reformer_model.eval()
    electras_model = ElectraModel.from_pretrained('google/electra-small-discriminator')
    electras_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electras_model = electras_model.to(device)
    electras_model.eval()
    
    scene_model = retinanet_resnet50_fpn_v2(pretrained=True)
    scene_model = scene_model.to(device)
    scene_model.eval()
    
    combined_features = []
    scene_feature_size= 1200

    for i in range(len(dataframe)):
        t_id = dataframe.tweet_id.iloc[i]
        path = images_path + str(t_id) + ".jpg"
        img = Image.open(path).convert("RGB")
        img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        convnext_image_features = convnext_model(img)
        convnext_image_features = convnext_image_features.view(convnext_image_features.size(0), -1)
        image_features = convnext_image_features.detach().cpu().numpy()

        text = dataframe.tweet_text.iloc[i]
        reformer_text_input = reformer_tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, padding='longest', return_tensors='pt', return_attention_mask=True)
        reformer_input_ids = reformer_text_input['input_ids'].to(device)
        reformer_attention_mask = reformer_text_input['attention_mask'].to(device)

        with torch.no_grad():
            reformer_text_features = reformer_model(input_ids=reformer_input_ids, attention_mask=reformer_attention_mask)[0][:, 0, :]
        reformer_text_features = reformer_text_features.detach().cpu().numpy()
        
        electras_text_tokens = electras_tokenizer.encode(text, add_special_tokens=True)
        electras_text_input = torch.tensor(electras_text_tokens).unsqueeze(0).to(device)
        electras_text_features = electras_model(electras_text_input)[0]
        electras_text_features = torch.mean(electras_text_features, dim=1).detach().cpu().numpy()
        text_features = np.concatenate((reformer_text_features, electras_text_features), axis=-1)

        with torch.no_grad():
            scene_outputs = scene_model(img)
        scene_features = scene_outputs[0]['boxes'].view(-1).detach().cpu().numpy()
        scene_features = scene_features[:scene_feature_size]
        while len(scene_features) < scene_feature_size:
            scene_features = np.append(scene_features, 0)
        scene_features = scene_features.reshape(1, -1)
       
        combined_features.append(np.concatenate((text_features, image_features,scene_features), axis=-1))
        
    combined_features = np.concatenate(combined_features, axis=0)
    scaler = MinMaxScaler()
    combined_features_transformed = scaler.fit_transform(combined_features)

    return combined_features_transformed