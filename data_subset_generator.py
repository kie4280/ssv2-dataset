from argparse import ArgumentParser, Namespace
from typing import List
import numpy as np
import math
import csv
import json


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument("input_path", type=str)
  parser.add_argument("label2idx_path", type=str)
  parser.add_argument(
      "--output",
      dest="output_path",
      type=str,
      default="annotations/output.json")
  parser.add_argument("--n_samples", type=int, required=True)
  parser.add_argument("--dataset_classes", type=int, default=174)
  parser.add_argument("--seed", type=int, default=123)
  return parser.parse_args()


def main(args: Namespace) -> None:
  np.random.seed(int(args.seed))
  with open(args.label2idx_path, "r") as f:
    label2idx = json.load(f)
  with open(args.input_path, "r") as f:
    samples = json.load(f)
  classes = [[] for _ in range(int(args.dataset_classes))]
  for idx, sample in enumerate(samples):
    template: str = sample["template"]
    template = template.replace("[", "").replace("]", "")
    classes[int(label2idx[template])].append(idx)

  n_samples = int(args.n_samples)
  sample_per_cls = math.floor(int(args.n_samples) / len(label2idx))
  collected_idx = []

  if sample_per_cls > 0:
    for s in classes:
      if len(s) > sample_per_cls:
        indexes = np.random.choice(len(s), sample_per_cls, replace=False)
        collected_idx.extend([s[i] for i in indexes])
        n_samples -= sample_per_cls
      else:
        collected_idx.extend(s)
        n_samples -= len(s)

  if n_samples > 0:
    assert n_samples < args.dataset_classes
    indexes = np.random.choice(
        int(args.dataset_classes),
        n_samples,
        replace=False)
    for index in indexes:
      collected_idx.append(classes[index][0])

  collected = []
  for idx in collected_idx:
    collected.append(samples[idx])
  json.dump(collected, open(args.output_path, "w"))


if __name__ == "__main__":
  args = parse_args()
  main(args)
