"""

5 - Lo Benchmark.py

This script takes the prediction data that Andrew Lo et al. 2019 made on our subsample of drugs and
generates an AUC curve to compare with our model results

"""

# +
## Managing Dependencies ##

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import scikitplot as skplt
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


# +
## Importing our data ##

drug_data = pd.read_csv("compounds_in_pipeline - copy.csv", encoding= 'unicode_escape')

## Drop last 
drug_data.drop(drug_data.tail(1).index,inplace=True)
# -

## Get columns for predictions and outcomes ##
predictions = drug_data['Probability'].to_numpy()
outcomes = drug_data['Test Variable'].to_numpy()

## Create AUC using predictive values ##
print("ROC AUC score:", roc_auc_score(outcomes, predictions))
print("Average precision score:", average_precision_score(outcomes, predictions))


# +
## Sci-kitplot ROC AUC ##
inverse_predictions = 1 - predictions
predictions_array = np.stack((1-predictions, predictions), axis = 1)

skplt.metrics.plot_roc_curve(outcomes, predictions_array, 
                             title = "ROC Curve from predictions of Lo et al. 2019", 
                             curves = 'macro',
                            cmap = 'Cyclic')
plt.show()

# +
## -----------------------------------------------    ##
