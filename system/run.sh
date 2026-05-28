#!/bin/bash
# Run all baseline experiments (8 algorithms x 2 datasets)
python main.py --config experiments/algorithm_comparison.json

# Or run a single default experiment:
# python main.py --config experiments/default.json
