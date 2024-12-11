## CS 6120 Project - Differential Transformers

### Sentiment Classification
Classification uses an encoder model for two different tasks
* Binary classification - SST-2 dataset: https://huggingface.co/datasets/stanfordnlp/sst2
  * The test set is from: https://huggingface.co/datasets/SetFit/sst2 (This is a smaller version of the dataset but it has about 95% the same test set - the larger set has the test labels hidden. The remaining ~5% of examples are NOT present in the train or development set, so they are safe to use (discussed with and cleared by professor during class))
* Multi-class (fine-grained) classification - SST-5 dataset: https://huggingface.co/datasets/SetFit/sst5

To run the classification task, you can run the classification.py file

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

### Recommended commands to run for classification:
* Differential/Binary:
  * `python classification.py --model DIFFERENTIAL --task BINARY --num_heads 1`
* Differential/Multi-class:
  * `python classification.py --model DIFFERENTIAL --task MULTICLASS --num_heads 1 --epochs 40`
* Traditional/Binary:
  * `python classification.py --model TRADITIONAL --task BINARY`
* Traditional/Multi-class:
  * `python classification.py --model DIFFERENTIAL --task MULTICLASS --epochs 40`

parser.add_argument('--model', type=str, default='DIFFERENTIAL', help='model to run (DIFFERENTIAL or TRADITIONAL)')
    parser.add_argument('--task', type=str, default='BINARY', help='classification task to run (BINARY OR MULTICLASS')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to run')
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