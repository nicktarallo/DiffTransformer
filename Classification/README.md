## Sentiment Classification
Classification uses an encoder model for two different tasks
* Binary classification - SST-2 dataset: https://huggingface.co/datasets/stanfordnlp/sst2
  * The test set is from: https://huggingface.co/datasets/SetFit/sst2 (This is a smaller version of the dataset but it has about 95% the same test set - the larger set has the test labels hidden. The remaining ~5% of examples are NOT present in the train or development set, so they are safe to use (discussed with and cleared by professor during class))
* Multi-class (fine-grained) classification - SST-5 dataset: https://huggingface.co/datasets/SetFit/sst5

To run the classification task, you can run the classification.py file to train a model.
Prior to running the command, be sure to be in the Classification directory. If you are not, do

```bash
cd Classification
```

within the larger project directory.

When running, it will train a model and make plots of relevant accuracy and F1 metrics at the end.
At the end of training, a csv with relevant training/development metrics will be saved to /df_outputs that can be used to do more advanced plotting and analysis.
Also at the end of training, the trained model will be saved to /models.

Libraries required: numpy, pandas, matplotlib, datasets, torch

These are the command line arguments you can use:
* --model DIFFERENTIAL or --model TRADITIONAL (default is DIFFERENTIAL)
  * This determines whether you train a differential or traditional transformer
* --task BINARY or --task MULTICLASS (default is BINARY)
  * This determines whether you train for the binary classifier or the fine-grain classifier
* --epochs [int] (default is 25)
  * This determines the number of epochs the model will train for
* --num_heads [int] (default is 2)
  * This determines the number of heads per layer (we used twice the amount for a traditional transformer compared to a differential one)
* --hidden_size [int] (default is 100)
  * This determines the hidden size of the FFNN in the transformer
* --transformer_layers [int] (default is 1)
  * This determines the number of layers in the transformer model
* --d_model [int] (default is 32)
  * This determines the embedding size
* --d_internal [int] (default is 32)
  * This determines the internal dimension used for Q, K, and V matrices
* --batch_size [int] (default is 32)
  * This determines the batch size for training
* --run_on_test (default is False)
  * Use this flag with no arguments when you want to run on the test set as well as validation at the end
* --get_attention_maps (default is False)
  * Use this flag with no arguments when you want to save example attention maps to the /maps folder

### Recommended commands to run for classification to fully train (Remember to enter Classification directory first):
* Differential/Binary:
  * `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 1 --run_on_test`
* Differential/Multi-class:
  * `python classification.py --model DIFFERENTIAL --task MULTICLASS --num_heads 1 --epochs 40 --run_on_test`
* Traditional/Binary:
  * `python classification.py --model TRADITIONAL --task BINARY --run_on_test`
* Traditional/Multi-class:
  * `python classification.py --model TRADITIONAL --task MULTICLASS --epochs 40 --run_on_test`

Epochs take a long time to run, so if you want to quickly see a plot you can just use `--epochs 2` to see the end result, although it won't be fully trained
