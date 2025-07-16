import torch
import pickle
corner_names = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
colors = ['blue', 'red', 'green']
description_list = [f'push the {color} block to the {corner} corner' for color in colors for corner in corner_names]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
description_embedding_dict = {description: torch.randn(1,512).to(device) for description in description_list}
#save the description_embedding_dict
with open('../../output/random_embeddings.pkl', 'wb') as f:
    pickle.dump(description_embedding_dict, f)