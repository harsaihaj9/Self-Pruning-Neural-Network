# Short Report

## Why L1 on sigmoid gates encourages sparsity
The sigmoid converts gate scores into values between 0 and 1. Adding an L1 penalty minimizes the sum of gate values. This pushes many gates toward 0, effectively removing unimportant connections while preserving useful ones needed for classification.

## Expected Trade-off
- Low lambda: better accuracy, lower sparsity
- High lambda: higher sparsity, possible accuracy drop

## Results Table
Run `python main.py` to generate actual results in `results.csv`.

## Plot
`gate_distribution.png` is generated after training.
