import cudf
import numpy as np

# Create a random DataFrame with cuDF
df = cudf.DataFrame({'a': np.random.rand(1000), 'b': np.random.rand(1000)})

print(df.head())
