# A Dynamic Speaker Model for Conversational Interactions

This repository includes the codes and models for the paper
[A Dynamic Speaker Model for Conversational Interactions](https://www.aclweb.org/anthology/N19-1284)
for reproducing the experiment results on Switchboard Dialog Act classification.
```
@InProceedings{Cheng2019NAACL,
  author    = {Hao Cheng and Hao Fang and Mari Ostendorf},
  title     = {A Dynamic Speaker Model for Conversational Interactions},
  booktitle = {Proc. Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year      = {2019},
  pages     = {2772--2785},
  url       = {https://www.aclweb.org/anthology/N19-1284},
}
```

## Requirements
* Python >= 2.7
* [virtualenv](https://virtualenv.pypa.io/en/latest/) 

## SWDA Data Processing
* Set up the Python virtual environment for data processing: 
    ```bash
    virtualenv env/venv_data_processing
    source env/venv_data_processing/bin/activate
    pip install -r data_script/requirement.txt
    ```
* Install `spacy==2.0.16` and download spacy model.
    ```bash
    pip install spacy==2.0.16
    python -m spacy download en_core_web_sm
    python -m spacy link en_core_web_sm en
    ```
* Install `nltk==2.0.5`. You may get the following error if you directly use `pip install nltk==2.0.5`.
	```
	urllib2.HTTPError: HTTP Error 403: SSL is required
	```
	* Download the package from [here](https://pypi.python.org/pypi/nltk/2.0.5).
	Untar the package, and edit `distribute_setup.py`: change `http` to `https` in line 50 
		```
		DEFAULT_URL = "https://pypi.python.org/packages/source/d/distribute/"
		```
	* Install the package using pip.
		```
		pip install ./nltk-2.0.5
		```
* Download the [swda.zip](http://compprag.christopherpotts.net/code-data/swda.zip) from [The Switchboard Dialog Act Corpus](http://compprag.christopherpotts.net/swda.html)
 and unzip it into `data/swda`.
* Run the data processing script 
   ```bash
   ./process_predictor_data.sh
   ```
   This script produces two subdirectories `data/swda_user_dialog_dir` and `data/swda_predictor_dialog_dir`
   which contain the converted data for training the dynamic speaker model and the dialog act tagging model, respectively.


## SWDA Dialog Act Tagging Model
* Set up the Python virtual environment for the tagging model.
    ```bash
    virtualenv env/venv_tagging_model
    source env/venv_tagging_model/bin/activate
    pip install -r src/requirement.txt
    ```
* Untar the pretrained model
    ```bash
    tar -xzvf swda_model.tgz
    ```
* Run the following script to evaluate the model 
    ```bash
    ./eval_tagger_model.sh
    ```
    This script outputs the evaluation results and dumps the prediction into `misc/eval_tagger_model_dir`.

* To train your own model, please see `./train_tagger_model.sh` for details.

## Train and Evaluate Dynamic Speaker Models
Please see `./train_user_model.sh` and `./eval_user_model.sh` for details.
