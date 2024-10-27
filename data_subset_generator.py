from argparse import ArgumentParser, Namespace
from typing import List
import numpy as np
import math
import csv
import json


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument("--input_path", type=str, required=True)
  parser.add_argument("--output_path", type=str, required=True)
  parser.add_argument("--n_samples", type=int, required=True)
  parser.add_argument("--dataset_classes", type=int, default=400)
  parser.add_argument("--seed", type=int, default=123)
  return parser.parse_args()


def main(args: Namespace) -> None:
  np.random.seed(int(args.seed))
  with open(args.input_path, "r") as f:
    df = json.load(f)
  classes = dict()
  for index, row in df.iterrows():
    cla: List = classes.get(row.label, [])
    cla.append(index)
    classes[row.label] = cla

  n_samples = int(args.n_samples)
  sample_per_cls = math.floor(int(args.n_samples) / len(classes))
  collected_idx = []

  if sample_per_cls > 0:
    for key, value in classes.items():
      if len(value) > sample_per_cls:
        indexes = np.random.choice(len(value), sample_per_cls, replace=False)
        collected_idx.extend([value[i] for i in indexes])
        n_samples -= sample_per_cls
      else:
        collected_idx.extend(value)
        n_samples -= len(value)

  if n_samples > 0:
    assert n_samples < args.dataset_classes
    indexes = np.random.choice(
        int(args.dataset_classes),
        n_samples,
        replace=False)
    class_list = list(classes.keys())
    for index in indexes:
      collected_idx.append(classes[class_list[index]][0])

  df_subset = df.iloc[collected_idx]
  df_subset.to_csv(args.output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
  args = parse_args()
  main(args)
