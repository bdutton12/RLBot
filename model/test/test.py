import numpy as np


batch_start = np.arange(0, 20, 5)
indices = np.arange(20, dtype=np.int64)
np.random.shuffle(indices)
batches = [indices[i:i+5] for i in batch_start]

test = np.arange(0, 20)

print(batches)
print(test[batches[0]])

