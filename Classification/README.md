# Sentiment Classification
The classification compares a differential and traditional encoder transformer for two different tasks:
* __Binary classification__: SST-2 dataset: https://huggingface.co/datasets/stanfordnlp/sst2
  * The test set is from: https://huggingface.co/datasets/SetFit/sst2 (This is a smaller version of the dataset but it has about 95% the same test set - the larger set has the test labels hidden. The remaining ~5% of examples are NOT present in the train or development set, so they are safe to use (discussed with and cleared by professor during class))
* __Multi-class (fine-grained) classification__: SST-5 dataset: https://huggingface.co/datasets/SetFit/sst5

## Installation
Libraries required: numpy, pandas, matplotlib, datasets, torch, scikit-learn.
If you do not have them, please run
```bash
pip install numpy pandas matplotlib datasets torch scikit-learn
```

## Running the code:
To run the classification task, you can run the classification.py file to train a model.

When running, it will train a model and make plots of relevant accuracy and F1 metrics at the end.
At the end of training, a csv with relevant training/development metrics will be saved to /df_outputs that can be used to do more advanced plotting and analysis.
Also at the end of training, the trained model will be saved to /models.

Prior to running the command, be sure to be in the Classification directory. If you are not, do

```bash
cd Classification
```

within the larger project directory.

The basic way to run the program is with
```bash
python classification.py
```
which will train a single-head differential transformer on binary classification for 25 epochs.

These are the command line arguments you can use:
* __--model DIFFERENTIAL__ or __--model TRADITIONAL__ (default is DIFFERENTIAL)
  * This determines whether you train a differential or traditional transformer
* __--task BINARY__ or __--task MULTICLASS__ (default is BINARY)
  * This determines whether you train for the binary classifier or the fine-grain classifier
* __--epochs [int]__ (default is 25)
  * This determines the number of epochs the model will train for
* __--num_heads [int]__ (default is 1)
  * This determines the number of heads per layer (we used twice the amount for a traditional transformer compared to a differential one)
* __--hidden_size [int]__ (default is 100)
  * This determines the hidden size of the FFNN in the transformer
* __--transformer_layers [int]__ (default is 1)
  * This determines the number of layers in the transformer model
* __--d_model [int]__ (default is 32)
  * This determines the embedding size
* __--d_internal [int]__ (default is 32)
  * This determines the internal dimension used for Q, K, and V matrices
* __--batch_size [int]__ (default is 32)
  * This determines the batch size for training
* __--run_on_test__ (default is False)
  * Use this flag with no arguments when you want to run on the test set as well as validation at the end
* __--get_attention_maps__ (default is False)
  * Use this flag with no arguments when you want to save example attention maps to the /maps folder

### Recommended commands to run for classification for testing (Remember to enter /Classification directory first):
* __Differential/Binary__:
  * `python classification.py --model DIFFERENTIAL --task BINARY --run_on_test`
* __Differential/Multi-class__:
  * `python classification.py --model DIFFERENTIAL --task MULTICLASS --epochs 60 --num_heads 8 --run_on_test`
* __Traditional/Binary__:
  * `python classification.py --model TRADITIONAL --task BINARY --num_heads 2 --run_on_test`
* __Traditional/Multi-class__:
  * `python classification.py --model TRADITIONAL --task MULTICLASS --epochs 60 --num_heads 16 --run_on_test`

In these four examples, the binary tasks are running on config B1-a from our report and the multi-class configs are running on config M2

Epochs take a long time to run, so if you want to quickly see a plot you can just use `--epochs 2` to see the end result

You can also just exclude all the additional parameters and just run on a basic 1 head, 1 layer, 100 FFNN size, and 25 epochs with these commands that will train faster:

* __Differential/Binary__:
  * `python classification.py --model DIFFERENTIAL --task BINARY --run_on_test`
* __Differential/Multi-class__:
  * `python classification.py --model DIFFERENTIAL --task MULTICLASS --run_on_test`
* __Traditional/Binary__:
  * `python classification.py --model TRADITIONAL --task BINARY --run_on_test`
* __Traditional/Multi-class__:
  * `python classification.py --model TRADITIONAL --task MULTICLASS --run_on_test`


## Outputs:
* Validation metrics will be outputted to the console at the end of training
* Test metrics will be outputted to the console if the --run_on_test flag is used
* Plots of accuracy and F1 will also be output

## Commands for replication of exact configurations from report (some of these will take a long time to train):
* __B1:__
  * __a:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 1 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task BINARY --num_heads 2 --run_on_test`
  * __b:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 2 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task BINARY --num_heads 4 --run_on_test`
  * __c:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 4 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task BINARY --num_heads 8 --run_on_test`
  * __d:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 8 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task BINARY --num_heads 16 --run_on_test`
* __M1:
  * __a:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 500 --epochs 100 --num_heads 1 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 500 --epochs 100 --num_heads 2 --run_on_test`
  * __b:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 500 --epochs 55 --num_heads 2 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 500 --epochs 55 --num_heads 4 --run_on_test`
  * __c:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 500 --epochs 50 --num_heads 4 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 500 --epochs 50 --num_heads 8 --run_on_test`
  * __d:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 500 --epochs 45 --num_heads 8 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 500 --epochs 45 --num_heads 16 --run_on_test`
* __M2:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 100 --epochs 60 --num_heads 8 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 100 --epochs 60 --num_heads 16 --run_on_test`
* __M3:__
    * __Differential__: `python classification.py --model DIFFERENTIAL --task MULTICLASS --hidden_size 500 --epochs 30 --num_heads 8 --transformer_layers 2 --run_on_test`
    * __Traditional__: `python classification.py --model TRADITIONAL --task MULTICLASS --hidden_size 500 --epochs 30 --num_heads 16 --transformer_layers 2 --run_on_test`