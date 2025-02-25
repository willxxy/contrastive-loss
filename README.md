# contrastive-loss
conditional contrastive loss in jax and pytorch inspired by https://arxiv.org/pdf/2210.01936
most difference is we dont have strong alternative img/captions as in the mentioned paper.

## Run
For jax (flax==0.10.2, jax[cuda12]==0.5.0)

`python contrast_loss_jax.py`

For torch (torch==2.2.0)

`python contrast_loss_torch.py`
