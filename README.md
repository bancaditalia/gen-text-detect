# Exploring Naive Approaches to Tell Apart LLMs Productions from Human-written Text

This is the official repository for code and results of the paper:

**Exploring Naive Approaches to Tell Apart LLMs Productions from Human-written Text**. Oliver Giudice, Alessandro Maggi, Matteo Nardelli, NLPIR 2023

Powerful Large Language Models (large LMs or LLMs) such as BERT and GPT are making the task of detecting machine-generated text more and more prominent and crucial to minimize threats posed by text generation models misuse. 
Nonetheless, only a limited number of efforts exist so far, which can be classified into simple classifiers, zero-shot approaches, and fine-tuned LMs. These approaches usually rely on LMs whose discrimination accuracy decreases as the size difference in favor of the generator model increases (hence, a detector should always employ a LM with at least the same number of parameters of the source LM).
Also, most of these approaches do not explicitly investigate whether the sentence syntactic structure can provide additional information that helps to build better detectors. 
All these considerations make the generalizing ability of detection methods into question. While generation techniques become more and more capable of producing human-like text, are the detection techniques capable of keeping up if not properly trained?
In this paper, we evaluate the most effective (and reproducible) detection method available in the state of the art in order to figure out the limits in its robustness. We complement this analysis by discussing results obtained using a novel naive approach that demonstrably achieves comparable results in terms of robustness with respect to much more advanced and sophisticated state-of-the-art methods.

---

## Setup 

Create a new environment using conda
``` bash 
conda env create -f env_transf291.yml
```

Activate and install `ipykernel`
```bash
conda activate transf291
conda install ipykernel
```

You should be able to use your conda environent as Jupyter Notebook kernels. 

In case you can't use transf291, you can try by adding it as kernel as follows. 
``` bash 
ipython kernel install --user --name=transf291
```

Start Jupyter Notebook: 
```bash 
jupyter notebook
```

---

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{giudice2023text,
   title={Exploring Naive Approaches to Tell Apart LLMs Productions from Human-written Text},
   author={Giudice, Oliver and Maggi, Alessandro and Nardelli, Matteo},
   booktitle={7th International Conference on Natural Language Processing and Information Retrieval},
   year={2023},
   organization = {ACM}
}
```
