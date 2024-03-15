import matplotlib.pyplot as plt
import re

RESULT_FILE = "training_log/dapg_relocate-mug-0.8_relocate-mug_0.1_100_trpo_seed200/results.txt"
#RESULT_FILE = "training_log/soil_relocate-large_clamp-0.8_relocate-large_clamp_0.01_100_trpo_seed100/results.txt"

with open(RESULT_FILE, 'r') as f:
  data = f.readlines()

data = [re.sub("\s+", " ", line.strip()).split(" ") for line in data[2:-1]]
data = [list(float(e) for e in lst) for lst in data]

i, y, y_eval, y_max = zip(*data)

plt.figure()
plt.plot(i, y, label="reward")
plt.plot(i, y_eval, label="eval reward")
plt.plot(i, y_max, label="max reward")
plt.xticks(list(range(0, int(i[-1]) + 1, 100)), rotation=45)
plt.yticks(list(range(int(min(y[0], y_eval[0])), int(max(y_eval[-1], y_max[-1])), 1000)))
plt.grid()
plt.legend()
plt.title(f"Reward {y[-1]}, Eval {y_eval[-1]}, Max {y_max[-1]}")
plt.savefig(f"{RESULT_FILE}.png")
