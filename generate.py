from model import OneLayerAttentionModel
from torch.nn import functional as F
import torch
import pandas as pd
import yaml
import random 
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

block_size = config['block_size']

def encode(input):
    encode_dict = {
        "+": 10,
        "=": 11
    }
    return torch.tensor([encode_dict.get(char) if char in encode_dict else int(char) for char in input], dtype=torch.int64).unsqueeze(0)

def decode(input):
    decode_dict = {
        10: "+",
        11: "="
    }
    return "".join([decode_dict.get(int(char), str(char)) for char in input])

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] # becomes (B, C)
        probs = F.softmax(logits, dim=-1) # (B, C)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

def create_test_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        num1 = random.randint(10000, 99999)
        num2 = random.randint(10000, 99999)
        result = num1 + num2
        data.append((num1, num2, result))
    return data

def evaluate(model, data):
    test_data = create_test_data()

    ground_truth = []
    predictions = []
    input_list = []
    for num1, num2, result in test_data:
        ground_truth.append(result)
        input = f"{num1}+{num2}="
        input_list.append(input)
        idx = encode(input)
        output = generate(model, idx, 6)
        result = decode(output[0].numpy())
        predictions.append(int(result.replace(input, "")))
    return input_list, ground_truth, predictions

def get_stats(input_list, ground_truth, predictions):
    num_correct = 0
    wrong_examples = []
    for input, gt, pred in zip(input_list, ground_truth, predictions):
        if gt == pred:
            num_correct += 1
        else:
            wrong_examples.append((input, gt, pred))
    return num_correct, wrong_examples

def save_df(input_list, ground_truth, predictions):
    df = pd.DataFrame({
        'input': input_list,
        'ground_truth': ground_truth,
        'predictions': predictions
    })
    df['correct'] = df['ground_truth'] == df['predictions']
    df.to_csv('results.csv', index=False)


def main():
    model = OneLayerAttentionModel()
    model.load_state_dict(torch.load('one_layer_attention_model.pth'))
    model.eval()

    input_list, ground_truth, predictions = evaluate(model, create_test_data())
    save_df(input_list, ground_truth, predictions)
    num_correct, wrong_examples = get_stats(input_list, ground_truth, predictions)
    print(f"Number of correct predictions: {num_correct}")
    print(f"Number of wrong predictions: {len(wrong_examples)}")
    print("Examples of wrong predictions:")
    for example in wrong_examples[:10]:
        print(example)

if __name__ == '__main__':
    main()