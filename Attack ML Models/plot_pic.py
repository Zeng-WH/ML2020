import matplotlib.pyplot as plt
import numpy as np

adv_examples = np.load('adv_examples.npy', allow_pickle=True)

data_raw = adv_examples[0][2]
adv = adv_examples[0][3]

data_raw = np.transpose(data_raw, (1,2,0))
adv = np.transpose(adv, (1,2,0))
plt.imshow(data_raw)
plt.show()
plt.imshow(adv)
plt.show()

print('test')