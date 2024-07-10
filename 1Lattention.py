import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import yaml
from model import OneLayerAttentionModel, data_generator
from viz import save_plot_epoch
from tqdm import tqdm 
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def loss_fn(logits, tokens, n_digits=5):
  ans_logits = logits[:, -(n_digits+2):-1]
  ans_probs = F.log_softmax(ans_logits.to(torch.float64), dim=-1)
  max_indices = torch.argmax(ans_probs, dim=-1)
  ans_tokens = tokens[:, -(n_digits+1):]
  ans_loss = torch.gather(ans_probs, -1, ans_tokens[:, :, None])[..., 0]
  per_token_train_losses = -ans_loss.mean(0)
  return per_token_train_losses.mean()

def main():
    batch_size = config['batch_size']
    eval_interval = config['eval_interval']
    learning_rate = config['learning_rate']
    n_digits = config['n_digits']
    weight_decay = config['weight_decay']
    n_epochs = config['n_epochs']
    ds = data_generator(batch_size, n_digits)
    ds_eval = data_generator(1024, n_digits, seed=12345)
    model = OneLayerAttentionModel()
    optimizer = optim.AdamW(model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))

    train_losses_list = []
    for epoch in tqdm(range(n_epochs)):

        tokens = next(ds)
        logits = model(tokens)

        train_loss = loss_fn(logits, tokens)
        train_loss.backward()
        train_losses_list.append(train_loss.item())

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if epoch % eval_interval == 0:
            with torch.no_grad():
                eval_tokens = next(ds_eval)
                logits = model(tokens)
                eval_loss = loss_fn(logits, tokens)
                torch.save(model.state_dict(), 'one_layer_attention_model.pth')
                save_plot_epoch(epoch)
            print(epoch, eval_loss.item())
    torch.save(model.state_dict(), 'one_layer_attention_model_final.pth')

if __name__ == "__main__":
    main()