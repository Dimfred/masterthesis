import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path


output = Path("imgs")

# plt.clf()

########################################################################################
## tuning yolo input size
########################################################################################

input_size = (544, 576, 608, 640, 672, 704, 736, 768, 800, 832)
valid = (92.534, 95.237, 96.370, 96.210, 95.659, 96.541, 97.006, 96.571, 96.359, 95.791)
test = (86.674, 87.695, 88.884, 91.531, 91.871, 92.227, 92.925, 93.506, 92.890, 92.885)

size = (3, 3)
fig = plt.figure(figsize=size)
both = [*valid, *test]
plt.xticks(input_size)
plt.yticks(range(int(min(both)), math.ceil(max(both))))

plt.plot(input_size, valid, "-o")
plt.plot(input_size, test, "-o")

plt.legend(("Validation", "Test"))
plt.show()
plt.savefig(output / "yolo_input_size_tuning.png")
plt.clf()

########################################################################################
## tuning unet input size
########################################################################################
