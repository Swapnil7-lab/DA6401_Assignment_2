# DA6401_Assignment_2

## Wandb Report Link:
[Wandb Report](https://api.wandb.ai/links/cs22m088-iit-madras/9ruon33n)

---

## Part A

### To Run the Code:

#### **Usage**
```bash
da6401_dl_2_partA.py
    [-h]
    [-wp WANDB_PROJECT]
    [-we WANDB_ENTITY]
    [-e EPOCHS]
    [-b BATCH_SIZE]
    [-o OPTIMIZER]
    [-lr LEARNING_RATE]
    [-a ACTIVATION]
    [-fm FILTER_MULTIPLIER]
    [-dp DROPOUT]
    [-fc DENSE_NEURONS]
    [-nf NO_FILTERS]
    [-bn BATCH_NORMALIZATION]
    [-da DATA_AUGMENTATION]
    -ks KERNEL_SIZE [KERNEL_SIZE ...]
```

#### **To run code in Colab**:
1. Load the dataset:
```bash
!wget 'https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
!unzip -q nature_12K.zip
```

2. After loading the dataset and mounting the `.py` file in Colab, run:
```bash
!python3 "/content/drive/My Drive/Colab Notebooks/da6401_dl_2_parta.py" -ks 3 3 3 3 3
```

This will run the code with default parameters:
```python
config = {
    'activation': 'LeakyRelu',
    'batch_norm': False,
    'batch_size': 64,
    'data_aug': True,
    'dropout': 0,
    'fc_neurons': 128,
    'filter_multiplier': 2,
    'kernel_sizes': [3, 3, 3, 3, 3],
    'learning_rate': 0.0005,
    'num_filters': 16,
    'optimizer': 'nadam'
}
```

#### **Wandb Integration**:
To enable Wandb logging, ensure you have the following in your script:
```python
import wandb

wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config)
```
Log metrics like this:
```python
wandb.log({"loss": loss, "accuracy": acc})
```

---

## Part B

### To Run the Code:

#### **To run code in Colab**:
1. Load the dataset:
```bash
!wget 'https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
!unzip -q nature_12K.zip
```

2. After loading the dataset and mounting the `.py` file in Colab, run:
```bash
!python3 "/content/drive/My Drive/Colab Notebooks/da6401_dl_2_partb.py"
```

---
