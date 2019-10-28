import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x': np.random.normal(-5, 3, 10000) # mean = -5 , standard deviation = 3
})

df.plot.kde()
plt.show()

print(np.mean(df['x']))
print(np.sqrt(np.sum((df['x']-np.mean(df['x'])) ** 2)/(len(df['x'])-1)))
print(np.sqrt(np.sum((df['x']-np.mean(df['x'])) ** 2)/(len(df['x']))))
print(np.std(df['x']))