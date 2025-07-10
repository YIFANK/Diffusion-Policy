#use CLIP to encode the description
from transformers import CLIPTokenizer, CLIPTextModel
import torch
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

corner_names = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

dataset_path = '../output/save_data/test_workspace.pkl'
path_list = [f'../dataset/{color}_{num}.pkl' for color in ['Blue', 'Red', 'Green'] for num in [0,1,2,3]]
description_list = [f'push the {color} block to the {corner} corner' for color in ['blue', 'red', 'green'] for corner in corner_names]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#encode the description
def encode_text(text):
    tokens = tokenizer(text = text, padding=True, return_tensors="pt")
    with torch.no_grad():
        return text_model(**tokens).last_hidden_state[:, 0, :].to(device) # (B, 512)

#make a dictionary of description and text embedding
description_embedding_dict = {description: encode_text(description) for description in description_list}
print(description_embedding_dict['push the blue block to the lower-right corner'].shape)
#check cosine similarity between two descriptions with 3 significant digits
cosine_sim = torch.cosine_similarity(description_embedding_dict['push the blue block to the lower-right corner'], description_embedding_dict['push the blue block to the lower-left corner'], dim=1)
print(f"Cosine similarity between 'push the blue block to the lower-right corner' and 'push the blue block to the lower-left corner': {cosine_sim.item():.3f}")
import pickle
#save the description_embedding_dict
with open('../../output/CLIP_embeddings.pkl', 'wb') as f:
    pickle.dump(description_embedding_dict, f)