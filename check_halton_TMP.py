
import json
from algorithmic_efficiency import halton

tuning_search_space = 'reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
num_tuning_trials = 3

with open(tuning_search_space, 'r', encoding='UTF-8') as search_space_file:
  tuning_search_space = halton.generate_search(
      json.load(search_space_file), num_tuning_trials)

tuning_search_space.sort()

for h in tuning_search_space:
  print(h[0])

# print(tuning_search_space)