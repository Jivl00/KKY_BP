import pandas as pd
import numpy as np
from IPython.display import display

device = 'laptop'
df = pd.read_json('temp/times_laptop.json')

for model in df:
    df[model]['model load'] = '{:.3f}+-{:.3f}'.format(np.mean(df[model]['model load']),
                                                      np.std(df[model]['model load']))
    df[model]['model inference'] = '{:.3f}+-{:.3f}'.format(np.mean(df[model]['model inference']),
                                                           np.std(df[model]['model inference']))
    df[str(model).replace('intent_classifier_', '')] = df[model]
    display(df[model])
    del df[model]

# display(df)