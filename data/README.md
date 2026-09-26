# Dataset

CardioLens-IoT uses the public **ECG Images Dataset of Cardiac Patients**:

> A. H. Khan and M. Hussain, "ECG Images dataset of Cardiac Patients," Mendeley Data, V2.
> doi: [10.17632/gwbz3fsgp8.2](https://doi.org/10.17632/gwbz3fsgp8.2)

The images are **not** included in this repository. Download them from the link above and check the dataset's license before reuse.

| Class | Images |
|---|---|
| Abnormal heartbeat | 233 |
| History of myocardial infarction | 172 |
| Myocardial infarction | 239 |
| Normal | 284 |
| **Total** | **928** |

## Local layout

Place the four class folders under `data/raw/` (their exact names do not matter as long as they contain the words *abnormal*, *history*, *myocardial/infarction* or *normal*):

```
data/raw/
├── ECG Images of Patient that have abnormal heartbeat/
├── ECG Images of Patient that have History of MI/
├── ECG Images of Myocardial Infarction Patients/
└── Normal Person ECG Images/
```

## Note on duplicates

The dataset contains byte-identical files. With the default split (seed 42), 121 training images have an exact copy in the validation or test set. `notebooks/benchmark_experiments.ipynb` also evaluates the model on a de-duplicated version.
