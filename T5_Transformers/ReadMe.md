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
* `T5_get_results.py` -- A script for loading the saved state dictionary of T5 and show several examples of predicted translations. **The code will not run right now because we did not submit the model due to size limit. 
* `T5_sample_results.txt` -- The file for outputting example translations by T5. **This is the output from running T5_get_results.py