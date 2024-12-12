import argparse
from datasets import load_dataset
import numpy as np
import torch
import random
from utils import *
from sklearn.metrics import f1_score, precision_score, recall_score
from traditional_encoder import Transformer
from differential_encoder import DiffTransformer
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# max_seq_length = 268  # This is the context length of the transformer
max_seq_length = 290


def _parse_args():
    """
    This specifies arguments that can be used on the command line to set the desired model
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--model', type=str, default='DIFFERENTIAL', help='model to run (DIFFERENTIAL or TRADITIONAL)')
    parser.add_argument('--task', type=str, default='BINARY', help='classification task to run (BINARY OR MULTICLASS')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to run')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads to use')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden size of feedforward layer')
    parser.add_argument('--transformer_layers', type=int, default=1, help='layers in the transformer')
    parser.add_argument('--d_model', type=int, default=32, help='embedding size')
    parser.add_argument('--d_internal', type=int, default=32, help='internal dimension size for Q, K, and V')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--run_on_test', dest='run_on_test', default=False, action='store_true',
                        help='use this flag to run on the test set')
    parser.add_argument('--get_attention_maps', dest='get_attention_maps', default=False, action='store_true',
                        help='use this flag to get attention maps')
    args = parser.parse_args()
    return args


def train_classifier(
        train: List[SentimentExample],
        validation: List[SentimentExample],
        indexer: Indexer,
        batch_size: int,
        model_type: str,
        task: str,
        num_epochs: int,
        num_heads: int,
        hidden_size: int,
        transformer_layers: int,
        d_model: int,
        d_internal: int,
        device: str = 'cpu'
) -> nn.Module:
    """
    Train a classifier
    :param train: The training examples
    :param validation: The validation examples
    :param indexer: The indexer
    :param batch_size: The size of batch for training
    :param model_type: Either "DIFFERENTIAL" or "TRADITIONAL"
    :param task: Either "BINARY" or "MULTICLASS"
    :param num_epochs: Number of epochs to train
    :param num_heads: Number of heads for multi-head attention
    :param hidden_size: Size of FFNN hidden layer in transformer
    :param transformer_layers: Number of layers in transformer
    :param d_model: Embedding size
    :param d_internal: Internal dimension for Q, K, V
    :param device: Device to run
    :return: the trained model
    """
    if task == 'BINARY':
        num_classes = 2
        f1_metrics = ['binary']  # For binary classification, use regular binary F1
    elif task == 'MULTICLASS':
        num_classes = 5
        f1_metrics = ['macro', 'weighted']  # For multiclass, use macro and weighted F1

    # Create the correct transformer type:
    if model_type == 'TRADITIONAL':
        model = Transformer(len(indexer), max_seq_length, d_model, d_internal, num_classes, transformer_layers,
                            num_heads, hidden_size).to(device)
    elif model_type == 'DIFFERENTIAL':
        model = DiffTransformer(len(indexer), max_seq_length, d_model, d_internal, num_classes, transformer_layers,
                                num_heads, hidden_size).to(device)

    # Adam optimizer with NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()

    model.train()
    accuracies = []
    losses = []
    f1_scores = {met: [] for met in f1_metrics}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Shuffle the training examples every epoch:
        random.shuffle(train)

        # Process the training data in batches:
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]
            # The mask is used to not attend to padding tokens when batching
            inputs, targets, mask = pad_batch(batch, indexer.index_of('<PAD>'), device=device)

            log_probs, _ = model.forward(inputs, mask)
            loss = loss_fcn(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()

        # Do an evaluation on the validation set every epoch for plotting at the end of training:
        num_correct, all_targets, all_predictions = run_examples(model, validation)

        # Calculate desired F1 scores and record for plotting:
        for metric, lst in f1_scores.items():
            f1 = f1_score(all_targets, all_predictions, average=metric)
            lst.append(f1)

        # Record accuracy and training loss for plotting:
        accuracies.append(num_correct / len(validation))
        print(f'Epoch {epoch + 1} loss: {epoch_loss}')
        losses.append(epoch_loss)

        model.train()


    model.eval()
    # Number of epochs
    epochs = range(1, len(accuracies) + 1)

    # Plotting the accuracy:
    plt.plot(epochs, accuracies, color='b', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the training loss:
    plt.plot(epochs, losses, color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Model Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting desired f1 metrics:
    for metric, lst in f1_scores.items():
        plt.plot(epochs, lst, color='b', label=f'{metric.capitalize()} F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.title('Model Validation F1 per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save useful metrics to a DataFrame:
    df = pd.DataFrame({'epoch': epochs, 'loss': losses, 'accuracy': accuracies,
                       **{f'{metric}_f1': scores for metric, scores in f1_scores.items()}})
    df.to_csv(f'df_outputs/{model_type}_{task}_{num_heads}head_df.csv', index=False)

    return model


def run_examples(model: nn.Module, exs: List[SentimentExample]) -> Tuple[int, List[int], List[int]]:
    """
    Run the provided model on the given examples and get the targets, outputs, and total number of correct predictions
    :param model: The model to run
    :param exs: The examples to test on
    :return: Number of correct answers, all target labels, and all predictions made by the model
    """
    num_correct = 0
    all_targets = []
    all_predictions = []

    # Make a prediction for each example and record the targets and predictions
    for example in exs:
        predicted = model.predict(example.input_tensor)
        if predicted == example.output:
            num_correct += 1
        all_targets.append(example.output)
        all_predictions.append(int(predicted))
    return num_correct, all_targets, all_predictions


def do_final_evaluation(model: nn.Module, exs: List[SentimentExample], f1_metrics: List[str]):
    """
    Print accuracy as well as all desired F1, precision, and recall metrics
    :param model: The trained model to evaluate
    :param exs: The examples to evaluate on
    :param f1_metrics: The average metrics to calculate for F1, precision, and recall
    :return: None
    """
    num_correct, all_targets, all_predictions = run_examples(model, exs)
    # Print accuracy
    print("accuracy:", num_correct / len(exs))

    # Print precision, recall, and F1 for all desired averaging metrics:
    for metric in f1_metrics:
        precision = precision_score(all_targets, all_predictions, average=metric)
        recall = recall_score(all_targets, all_predictions, average=metric)
        f1 = f1_score(all_targets, all_predictions, average=metric)
        print(f'{metric} precision: {precision}')
        print(f'{metric} recall: {recall}')
        print(f'{metric} F1: {f1}')


def get_attention_maps(model: nn.Module, example: SentimentExample, model_type: str, task: str, num_heads: int):
    """
    Plot and save attention maps for a single example
    :param model: The model to get the attention maps from
    :param example: The example to get the maps for
    :param model_type: One of "DIFFERENTIAL" or "TRADITIONAL" (used for file naming)
    :param task: One of "BINARY" or "MULTICLASS" (used for file naming)
    :param num_heads: Number of heads for multihead attention (used for file naming)
    :return: None
    """
    model.eval()
    # Get attention maps by running an example through the model:
    result, attn_maps = model.forward(example.input_tensor)

    # Since there may be more than one attention map for more than one head/layer, iterate through all
    for i in range(0, len(attn_maps)):
        attn_map = attn_maps[i]
        fig, ax = plt.subplots()
        # print(attn_map.size())
        map_np = attn_map.detach().cpu().numpy()
        # Use a heatmap to show attention
        im = ax.imshow(map_np.reshape(map_np.shape[1], map_np.shape[2]), cmap='hot', interpolation='nearest')
        # Put token labels on the x and y axes
        ax.set_xticks(np.arange(len(example.input.split(" "))))
        ax.set_xticklabels(example.input.split(" "), rotation=90)
        ax.set_yticks(np.arange(len(example.input.split(" "))))
        ax.set_yticklabels(example.input.split(" "))
        ax.xaxis.tick_top()
        # Display and save the map:
        # plt.show()
        plt.savefig(f'maps/{model_type}_{task}_{num_heads}head_attn_map_{i}.png')


def main():
    """
    Read arguments, train, and evaluate a model for classification as necessary
    :return: None
    """
    args = _parse_args()
    if args.task == 'BINARY':
        ds = load_dataset("stanfordnlp/sst2")  # Use the binary SST set
        # This dataset has the test labels unhidden, so we use it for the test dataset
        # It has almost all the same test examples, and the ones that don't match do not overlap
        # with the train or development set that we use
        # The larger one from stanfordnlp has the test labels hidden
        ds_test = load_dataset("SetFit/sst2").rename_column('text', 'sentence')
        f1_metrics = ['binary']  # F1 averaging strategy - binary means don't average, use class 1
    elif args.task == 'MULTICLASS':
        ds = ds_test = load_dataset('SetFit/sst5').rename_column('text', 'sentence')  # Use the 5-class SST set
        f1_metrics = ['macro', 'weighted']  # Multiclass F1 averaging strategy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ds['train']
    validation_dataset = ds["validation"]
    test_dataset = ds_test['test']

    # Create indexer and add tokens for unknown and pad:
    indexer = Indexer()
    indexer.add_and_get_index('<UNK>')
    indexer.add_and_get_index('<PAD>')

    # Add all tokens from the training dataset to the indexer:
    for example in train_dataset:
        for token in example['sentence'].split(' '):
            indexer.add_and_get_index(token)

    # Create SentimentExample objects for each section of the dataset (this will create tensors of indices)
    train_exs = [SentimentExample(ex['sentence'], ex['label'], indexer, device) for ex in train_dataset]
    validation_exs = [SentimentExample(ex['sentence'], ex['label'], indexer, device) for ex in validation_dataset]
    test_exs = [SentimentExample(ex['sentence'], ex['label'], indexer, device) for ex in test_dataset]

    # Train the model:
    model = train_classifier(
        train_exs,
        validation_exs,
        indexer,
        batch_size=args.batch_size,
        model_type=args.model,
        task=args.task,
        num_epochs=args.epochs,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        transformer_layers=args.transformer_layers,
        d_model=args.d_model,
        d_internal=args.d_internal,
        device=device,
    )
    torch.save(model, f'models/{args.model}_{args.task}_{args.num_heads}head_model.pt')

    # Print results on validation set
    print('Validation Results:')
    do_final_evaluation(model, validation_exs, f1_metrics)

    # Print results on test set if desired:
    if args.run_on_test:
        print()
        print('Test Results:')
        do_final_evaluation(model, test_exs, f1_metrics)

    # Display and save attention maps if desired:
    if args.get_attention_maps:
        get_attention_maps(model, validation_exs[0], args.model, args.task, args.num_heads)


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()
