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
    total_score = 0
    for color in colors[:1]:
        for num in range(4):
            corner = corner_names[num]
            description = f'push the {color} block to the {corner} corner'
            print(f'evaluating task: {description}')
            total_score += evaluate(model_path = model_path, policy_type = policy_type, task = [color, num],noise_pred_net_type = noise_pred_net_type)
    print(f'total score: {total_score} out of {len(colors)*4 * 10}')
    #save the total score to a file
    model_name = model_path.split('/')[-1].split('.')[0]
    with open(f'../output/eval/{model_name}/total_score.txt', 'w') as f:
        f.write(f'total score: {total_score} out of {len(colors)*4 * 10}')


from concept_inference import infer_new_concepts
def eval_with_learned_concept_weights(model_path: str = '../output/diffusion_policy_push.pth',
                                       policy_type: str = 'diffusion',init_type: str = 'original',noise_pred_net_type: str = 'unet',logging = False,K = 1,num_trajectories = 10):
    #load the learned concept weights
    #eval the model with the learned concept weights
    total_score = 0
    for idx in range(0,1):
        color = colors[idx]
        for num in range(4):
            corner = corner_names[num]
            description = f'push the {color} block to the {corner} corner'
            new_concept_dataset_path = f'../dataset/{Colors[idx]}_{num}.pkl'
            print(f'evaluating task: {description}, concept path: {new_concept_dataset_path}')
            learned_concepts = infer_new_concepts(model_path = model_path, policy_type = policy_type, task = [color, num],
            new_concept_dataset_path = new_concept_dataset_path, num_epochs = 500,learning_rate = 0.01,
            init_type = init_type,logging = logging,noise_pred_net_type = noise_pred_net_type,K = K,
            num_trajectories = num_trajectories)
            # weights = learned_concepts['weights']
            # embeddings = learned_concepts['embeddings']
            weights = learned_concepts['weights']
            embeddings = learned_concepts['embeddings']
            #eval the model with the learned concept weights
            total_score += evaluate(model_path = model_path, policy_type = policy_type, task = [color, num], concept_weights = weights, concept_embeddings = embeddings,
            init_type = init_type,noise_pred_net_type = noise_pred_net_type)
    print(f'total score: {total_score} out of {len(colors)*4 * 10}')
def count_total_score():
    #count the total score of ../output/eval/task_name/policy_type/init_type/all_episodes.gif
    task_list = [f'{color}{num}' for color in colors for num in range(4)]
    policy_type = 'flow_matching'
    init_type = 'original'
    total_score = 0
    for task_name in task_list:
        for file in os.listdir(f'../output/eval/{task_name}/{policy_type}/{init_type}'):
            if file.endswith('.txt'):
                with open(f'../output/eval/{task_name}/{policy_type}/{init_type}/{file}', 'r') as f:
                    #find the line that contains 'Total score:'
                    for line in f:
                        if 'Total score:' in line:
                            score = float(line.split('Total score:')[1].split('over')[0])
                            print(f'{task_name} score: {score}')
                            total_score += score
                            break
    print(f'total score: {total_score} out of {len(task_list) * 10}')
if __name__ == '__main__':
    # eval_with_original_concept_weights(model_path = '../trained_models/large_diffusion_policy_random.pth',
    #                                 policy_type = 'diffusion',noise_pred_net_type = 'unet')
    # count_total_score()
    # eval_with_original_concept_weights(model_path = '../trained_models/small_flow_policy_CLIP.pth',
    # policy_type = 'flow_matching',noise_pred_net_type = 'unet')
    # eval_with_learned_concept_weights(model_path = '../output/large_flow_policy.pth',
    #                                    policy_type = 'flow_matching',init_type = 'fixed')
    # eval_with_learned_concept_weights(model_path = '../trained_models/large_diffusion_policy_random.pth',
    #                                 policy_type = 'diffusion',init_type = 'rand',noise_pred_net_type = 'unet',logging = True)
    # eval_with_original_concept_weights(model_path = '../trained_models/large_flow_policy_random.pth',
    #                                 policy_type = 'flow_matching',noise_pred_net_type = 'unet')
    model_path = '../trained_models/VLM2Vec_LR.pth'
    model_name = model_path.split('/')[-1].split('.')[0]
    print(f"Model name: {model_name}")
    eval_with_original_concept_weights(model_path = model_path,
                                     policy_type = 'diffusion',noise_pred_net_type = 'unet')
    # eval_with_learned_concept_weights(model_path = model_path,
    #                                  policy_type = 'diffusion',init_type = 'simplex',noise_pred_net_type = 'unet',logging = True,K = 1,num_trajectories = None)
    # eval_with_learned_concept_weights(model_path = '../trained_models/large_flow_policy_random.pth',
    #                                 policy_type = 'flow_matching',init_type = 'fixed',noise_pred_net_type = 'unet',logging = True,K = 1)
