#auto-eval the model with different original concepts

import os
import numpy as np
import torch
from inference import evaluate
colors = ['blue', 'red', 'green']
Colors = ['Blue', 'Red', 'Green']
corner_names = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
def eval_with_original_concept_weights(model_path: str = '../output/diffusion_policy_push.pth',
                                       policy_type: str = 'diffusion',noise_pred_net_type: str = 'unet'):
    #load the original concept weights
    #eval the model with the original concept weights
    for color in colors:
        for num in range(4):
            corner = corner_names[num]
            description = f'push the {color} block to the {corner} corner'
            print(f'evaluating task: {description}')
            evaluate(model_path = model_path, policy_type = policy_type, task = [color, num],noise_pred_net_type = noise_pred_net_type)

from concept_inference import infer_new_concepts
def eval_with_learned_concept_weights(model_path: str = '../output/diffusion_policy_push.pth',
                                       policy_type: str = 'diffusion',init_type: str = 'original',noise_pred_net_type: str = 'unet',logging = False):
    #load the learned concept weights
    #eval the model with the learned concept weights
    for idx in range(1):
        color = colors[idx]
        for num in range(4):
            corner = corner_names[num]
            description = f'push the {color} block to the {corner} corner'
            new_concept_dataset_path = f'../dataset/{Colors[idx]}_{num}.pkl'
            print(f'evaluating task: {description}, concept path: {new_concept_dataset_path}')
            learned_concepts = infer_new_concepts(model_path = model_path, policy_type = policy_type, task = [color, num],
            new_concept_dataset_path = new_concept_dataset_path, num_epochs = 500,learning_rate = 1e-4,init_type = init_type,logging = logging,noise_pred_net_type = noise_pred_net_type)
            weights = learned_concepts['weights']
            embeddings = learned_concepts['embeddings']
            #eval the model with the learned concept weights
            evaluate(model_path = model_path, policy_type = policy_type, task = [color, num], concept_weights = weights, concept_embeddings = embeddings,
            init_type = init_type,noise_pred_net_type = noise_pred_net_type)

if __name__ == '__main__':
    # eval_with_original_concept_weights(model_path = '../trained_models/small_diffusion_policy_VLM2Vec.pth',
    #                                 policy_type = 'diffusion',noise_pred_net_type = 'unet')
    # eval_with_original_concept_weights(model_path = '../trained_models/small_diffusion_policy_VLM2Vec.pth',
    # policy_type = 'diffusion',noise_pred_net_type = 'unet')
    # eval_with_learned_concept_weights(model_path = '../output/large_flow_policy.pth',
    #                                    policy_type = 'flow_matching',init_type = 'fixed')
    # eval_with_learned_concept_weights(model_path = '../trained_models/diffusion_policy_VLM2Vec_transformer.pth',
    #                                    policy_type = 'diffusion',init_type = 'fixed',noise_pred_net_type = 'transformer')
    eval_with_learned_concept_weights(model_path = '../trained_models/small_diffusion_policy_VLM2Vec.pth',
                                       policy_type = 'diffusion',init_type = 'rand',logging = True)
