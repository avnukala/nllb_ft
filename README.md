## Enhancing Korean-English Machine Translation of Domain Text with Context-Based Transfer Learning

To create the conda environment for this repository, please execute the following command in your terminal

`conda env create --name envname --file=environments.yml`

The model training code is contained in `nllb.py` and includes a variety of options for training. For help determine what command line arguments to use, please run:

`python3 nllb.py --help`

The command line options are:
```
  --batch_size          batch training size
  --max_length          max sentence length
  --n_iters             total number of training iterations 
                        to use
  --print_every         print loss info every this many training 
                        examples
  --checkpoint_every    write out checkpoint every this many 
                        training examples
  --plot_every          plot this many training examples
  --initial_learning_rate 
                        initial learning rate
  --train_path          training file
  --val_path            validation/test file
  --out_file            output file
  --model_save_path     folder to save models
  --plot_save_path      folder to save plots
  --load_checkpoint     checkpoint file to start from
  --curriculum, -c      curriculum learning (process data by size)
  --lr                  learning rate
  --test_small, -ts     test bleu score on a smaller set
```