# gen-text-detect
Work on detecting synthetic text generated with deep neural language models (Stanford NLU project)


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
