"""Script to generate configs for MLP grid search."""

# Loop through all permuations of:
#   activation: (Relu(), Silu(), SwiGLU())
#   num_layers: range(1, 3)
#   expansion_ratio: range(2.0, 8.0) in 2 step size
#   dropout: range(0, 0.5) maybe 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5 (logarithmic)
#   normalization: True/Flase (layernorm, always)
#   lr: range(1e-3, 5e-5) do step of 5e-(+1) (obviously logarithmic)
#   weight_decay: range() 0.3, 0.1, 0.03, 0.01, 0.0 (logarithmic)
#   optimizer: AdamW()
#   batch_size: range()      
#       for batch size do a scan (?), batch_size vs inference_time plot: at what moment the plot becomes linear is optimal batch_size (not flat, but linear (x=y))
#       test batch size independently maybe. !!! DO THIS FIRST !!!
#   num_epochs: 20 (use non-loss metrics for early stopping: losses are super noisy. Use R^2 or kendall or AUROC. Use main eval metrics)

# Some of these can be cut down slightly

# 3*3*4*8*2*6*5 = 17,280 experiments
# 3*3*4*3*2*6*3 = 3,888 experiments
# 3*3*4*3*1*4*3 = 1,296 experiments
