from model import OneLayerAttentionModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

def plot_attention(attention, i, epoch):
    # Define a custom colormap from white to red
    colors = [[1, 1, 1], [0, 0, 0]]  # white to red
    colors[1][i%3] = 1
    new_cmap = LinearSegmentedColormap.from_list('white_to_red', colors, N=256)
    plt.figure(figsize=(4, 4))
    sns.heatmap(attention[0].detach().numpy(), cmap=new_cmap, cbar=True)
    # save the plot
    plt.savefig(f'./graphs/attention_head_{i}_{epoch}.png')
    plt.close()

def get_model_weights():
    model = OneLayerAttentionModel()
    model.load_state_dict(torch.load('one_layer_attention_model.pth'))
    token_embedding_weights = model.token_embedding_table.weight
    position_embedding_weights = model.position_embedding_table.weight
    heads = model.attention.sa.heads
    return token_embedding_weights, position_embedding_weights, heads

def calculate_attention(input, epoch, seq_len=18):
    token_embedding_weights, position_embedding_weights, heads = get_model_weights()
    token_embedding = token_embedding_weights[input]
    position_embedding = position_embedding_weights[torch.arange(input.size(1))]
    x = token_embedding + position_embedding
    tril = torch.tril(torch.ones(seq_len, seq_len))
    for i, head in enumerate(heads):
        key = head.key(x)
        query = head.query(x)
        attention = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
        attention = attention.masked_fill(tril[:seq_len, :seq_len] == 0, float('-inf'))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        plot_attention(attention, i, epoch)

def save_plot_epoch(epoch):
    input = torch.tensor([1, 2, 3, 4, 5, 10, 1, 1, 1, 1, 1, 11, 0, 2, 3, 4, 5, 6]).unsqueeze(0)
    attention = calculate_attention(input, epoch)        


