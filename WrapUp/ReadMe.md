## Dependencies 

* `pandas`
* `sentencepiece`
* `torch` with CUDA support
* `cudatoolkit`
* `transformers`
* `tensorboard`

To install, 

```
$ pip install pandas sentencepiece transformers tensorboard
$ pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```


## Description of each file 

* `T5.py` -- A script for fine-tuning pretrained T5 transformer model for Tibetan-English translation. 
* `T5_checkpoint_best_epoch=44.pt` -- The state dictionary of the T5 model with the lowest validation loss during 50 epochs. 
* `T5_get_results.py` -- A script for loading the saved state dictionary of T5 and show several examples of predicted translations. 
* `T5_sample_results.txt` -- The file for outputting example translations by T5. 

* `Scratch.py` -- A script for building and training a transformer from scratch for Tibetan-English-translation. 
* `Scratch_checkpoint_best_epoch=34.pt` -- The state dictionary of the transformer from scratch with the lowest validation loss during 50 epochs. 
* `Scratch_get_results.py` -- A script for loading the saved state dictionary of transformer from scratch and show several examples of predicted translations. 
* `Scratch_sample_results.txt` -- The file for outputting example translations by transformer from scratch. 

* `tensorboard_log` -- The directory that contains logged information such as training loss, validation loss, learning rate, etc. created by Tensorboard. To plot these information with Tensorboard, run `$ tensorboard --logdir ./tensorborard_log/` and visit the URL prompted in the command line. 