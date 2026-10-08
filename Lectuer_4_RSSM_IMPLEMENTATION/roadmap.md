# Lecture 4 Implementation Roadmap

Follow this roadmap in order. Do not start by building the complete RSSM. First reproduce the provided implementation, then understand each component through controlled experiments.

## Phase 0: Prepare the workspace

### Tasks

1. Clone the repository.
2. Locate the Lecture 3 code, Lecture 4 code, notebooks, slides, and dataset instructions.
3. Create a separate working directory for your own implementation.
4. Run the provided Lecture 4 notebook or script without modifications.

```bash
git clone https://github.com/RajatDandekar/build-a-world-model-from-scratch.git
cd build-a-world-model-from-scratch
```

### Deliverable

Create:

```text
lecture4_notes.md
```

Record:

- File names.
- Purpose of each file.
- Dataset location.
- Model dimensions.
- Training command.
- Evaluation command.
- Output files and plots.

Do not rewrite anything yet.

***

# Phase 1: Understand the robotics dataset

## Goal

Understand exactly what one training sample contains.

The dataset consists of robot trajectories containing camera observations and six-dimensional actions. Lecture 4 uses \(64 \times 64 \times 3\) images and six motor-action values. [youtube](https://www.youtube.com/watch?v=gNwczJjm-8o)

### Tasks

1. Load one episode.
2. Print the image shape.
3. Print the action shape.
4. Display frames in temporal order.
5. Plot the six action dimensions.
6. Identify the transition convention.

You must determine whether the data represents:

```text
image[t] + action[t] → image[t+1]
```

or:

```text
image[t+1] + action[t] → image[t]
```

### Deliverable

Create:

```text
01_dataset_inspection.ipynb
```

Include:

- Episode visualization.
- Action plots.
- Image normalization check.
- Train/validation split.
- Tensor-shape documentation.

Expected conceptual shapes:

```text
images:  [batch, time, channels, height, width]
actions: [batch, time, 6]
```

***

# Phase 2: Reproduce Lecture 3 on robotics data

## Goal

Understand the baseline that Lecture 4 improves.

Implement or run the deterministic model:

```text
image
  ↓
CNN encoder
  ↓
visual embedding
  +
action
  ↓
GRU
  ↓
deterministic hidden state h
  ↓
decoder
  ↓
predicted image
```

Mathematically:

\[
h_t=\operatorname{GRU}([e(o_t),a_t],h_{t-1})
\]

\[
\hat{o}_{t+1}=\operatorname{Decoder}(h_t)
\]

### Tasks

1. Train the deterministic model using the provided code.
2. Reconstruct or predict frames with teacher forcing.
3. Run an autonomous rollout.
4. Compare the predicted frame with the actual frame.
5. Measure error at several horizons.

Evaluate:

```text
1 step
5 steps
10 steps
20 steps
40 steps
60 steps
```

### Deliverables

```text
02_deterministic_baseline.ipynb
deterministic_baseline.pt
deterministic_rollout.mp4
deterministic_error_plot.png
```

### Understanding checkpoint

You should be able to explain:

- Why the model can remember movement.
- Why it struggles when the cube becomes occluded.
- Why uncertain futures produce averaged or blurry predictions.
- Why small errors compound during autonomous rollout.

Do not proceed until you can reproduce the deterministic baseline behavior.

***

# Phase 3: Understand the RSSM state

## Goal

Understand the two-part belief state before implementing the complete training loop.

The RSSM state is:

\[
b_t=[h_t,s_t]
\]

where:

- \(h_t\) is deterministic recurrent memory.
- \(s_t\) is stochastic latent state.

The lecture presents the combined state as approximately \(256+32=288\) dimensions. [youtube](https://www.youtube.com/watch?v=gNwczJjm-8o)

### Create a state diagram

```text
Previous stochastic state s_(t-1)
              +
Action a_(t-1)
              ↓
           GRU
              ↓
Deterministic state h_t
       ┌──────┴──────┐
       ↓             ↓
   Prior p(s_t)   Posterior q(s_t | o_t)
       │             │
       └──────┬──────┘
              ↓
         Sample s_t
              ↓
       Belief [h_t, s_t]
              ↓
           Decoder
              ↓
        Reconstructed image
```

### Tasks

Implement only the state transition first:

```python
h_t = GRU([s_prev, action], h_prev)
```

Then implement:

```python
prior(h_t) → prior_mean, prior_std
posterior(h_t, image_embedding) → post_mean, post_std
```

Test everything with random tensors.

### Deliverable

```text
03_rssm_shapes_and_states.ipynb
```

Verify:

```text
h_t shape
s_t shape
prior mean shape
prior std shape
posterior mean shape
posterior std shape
belief shape
```

***

# Phase 4: Implement the stochastic latent

## Goal

Understand how uncertainty enters the model.

Use the reparameterization trick:

\[
s_t=\mu_t+\sigma_t\odot\epsilon
\]

\[
\epsilon\sim\mathcal{N}(0,I)
\]

### Tasks

1. Implement the prior head.
2. Implement the posterior head.
3. Implement standard deviation using a positive function such as `softplus`.
4. Implement Gaussian sampling.
5. Check that repeated samples differ.
6. Check that gradients flow through the sampled state.

### Deliverable

```text
04_stochastic_latent_test.ipynb
```

You should be able to show:

```text
same mean/std + different noise → different s_t
```

but:

```text
same input + deterministic evaluation → stable output
```

***

# Phase 5: Build the posterior reconstruction model

## Goal

First make the model reconstruct observations using the posterior.

At this stage, the model is allowed to see the current image:

```text
image o_t
   ↓
CNN encoder
   ↓
image embedding
   +
h_t
   ↓
posterior q(s_t | h_t, o_t)
   ↓
sample s_t
   +
h_t
   ↓
decoder
   ↓
reconstructed image
```

### Tasks

1. Implement the image encoder.
2. Implement the posterior network.
3. Sample \(s_t\).
4. Concatenate \(h_t\) and \(s_t\).
5. Decode the belief state.
6. Train using reconstruction loss only initially.

\[
\mathcal{L}_{recon}
=
\|o_t-\hat{o}_t\|^2
\]

### Deliverable

```text
05_posterior_reconstruction.ipynb
posterior_reconstruction.pt
posterior_reconstructions.png
```

### Checkpoint

The model should reconstruct:

- Robot arm.
- Table.
- Cube.
- Gripper.
- Scene layout.

Do not focus on autonomous prediction yet.

***

# Phase 6: Add prior–posterior KL training

## Goal

Teach the prior to predict the stochastic state without seeing the future image.

The two distributions are:

\[
p(s_t\mid h_t)
\]

and:

\[
q(s_t\mid h_t,o_t)
\]

Train them with:

\[
\mathcal{L}_{KL}
=
D_{KL}
\left[
q(s_t\mid h_t,o_t)
\;\|\;
p(s_t\mid h_t)
\right]
\]

The prior is blind to the current image. The posterior sees the image and provides a better training target. The KL loss brings the prior closer to the posterior. [youtube](https://www.youtube.com/watch?v=gNwczJjm-8o)

### Total loss

\[
\mathcal{L}
=
\mathcal{L}_{recon}
+
\beta\mathcal{L}_{KL}
\]

Start with KL warm-up:

```python
beta = gradually_increase_beta()
```

### Tasks

1. Add the prior network.
2. Calculate prior mean and standard deviation.
3. Calculate posterior mean and standard deviation.
4. Calculate diagonal Gaussian KL divergence.
5. Log reconstruction and KL losses separately.
6. Check whether either distribution collapses.

### Deliverable

```text
06_prior_posterior_training.ipynb
rssm_posterior_model.pt
loss_curves.png
prior_posterior_statistics.png
```

### Monitor

```text
reconstruction loss
KL loss
prior mean
posterior mean
prior standard deviation
posterior standard deviation
hidden-state norm
stochastic-state norm
```

***

# Phase 7: Separate observe mode and imagine mode

## Goal

Implement the most important practical distinction in Lecture 4.

## Observe mode

Used when camera images are available:

```text
previous h, previous s, action, image
                    ↓
                  GRU
                    ↓
                  h_t
                    ↓
        prior and posterior distributions
                    ↓
       posterior sample s_t
                    ↓
              reconstructed image
```

## Imagine mode

Used when the camera is switched off:

```text
previous h, previous s, action
              ↓
             GRU
              ↓
             h_t
              ↓
      prior p(s_t | h_t)
              ↓
        sample s_t
              ↓
        decode [h_t, s_t]
```

Create two explicit functions:

```python
observe_step(h, s, action, image)
imagine_step(h, s, action)
```

Do not combine these into one unclear function.

### Deliverable

```text
07_observe_and_imagine.ipynb
```

Test that:

- `observe_step` uses the image.
- `imagine_step` does not use the image.
- Both return compatible \(h_t\), \(s_t\), and belief states.

***

# Phase 8: Implement camera-off dreaming

## Goal

Generate a future trajectory without future camera observations.

### Procedure

Use the first part of a real trajectory as context:

```text
real observations: o_0, o_1, ..., o_K
real actions:      a_0, a_1, ..., a_T
```

### Context phase

```python
for t in range(K):
    h, s = observe_step(
        h,
        s,
        actions[t],
        images[t],
    )
```

### Dream phase

```python
for t in range(K, T):
    h, s = imagine_step(
        h,
        s,
        actions[t],
    )

    frame = decoder(torch.cat([h, s], dim=-1))
```

After time \(K\), do not use:

- Real images.
- Real image embeddings.
- Posterior states.
- Ground-truth future latent states.

### Deliverable

```text
08_camera_off_dream.ipynb
rssm_final.pt
rssm_dream_rollout.mp4
```

Mark the transition in the video:

```text
frames before cutoff: real observations
frames after cutoff: model imagination
```

***

# Phase 9: Reproduce the three-model comparison

## Goal

Understand why RSSM is needed.

Compare:

### Model A: Deterministic

```text
h_t only
```

Expected problem:

```text
uncertain futures → averaged predictions → blur and drift
```

### Model B: Pure stochastic

```text
s_t only or independently sampled latent
```

Expected problem:

```text
uncertainty exists → persistent temporal memory is weak
```

### Model C: RSSM

```text
[h_t, s_t]
```

Expected advantage:

```text
persistent memory + sampled future branch
```

The lecture reports representative long-horizon errors of approximately \(0.409\) for the deterministic model, \(2.500\) for the pure stochastic model, and \(0.009\) for RSSM. Use these as reference values, not guaranteed results for your implementation. [youtube](https://www.youtube.com/watch?v=gNwczJjm-8o)

### Deliverable

Create:

```text
09_model_comparison.ipynb
model_comparison_table.csv
model_comparison_plot.png
```

Use the same:

- Initial context.
- Action sequence.
- Test episodes.
- Rollout horizon.
- Error metric.

***

# Phase 10: Reproduce the lecture analyses

## A. Linear probe

Test whether \(h_t\) contains physical robot information.

Procedure:

1. Freeze the RSSM.
2. Collect \(h_t\) over trajectories.
3. Collect actual joint positions or actions.
4. Fit a linear regression.
5. Compare predicted and true joint values.

```text
h_t → linear layer → six robot variables
```

### Deliverable

```text
linear_probe_results.png
```

Interpretation:

- Good prediction means \(h_t\) contains useful physical information.
- Poor prediction may indicate insufficient training or an incorrect data pipeline.

## B. Deterministic-state heatmap

Plot the recurrent state over time:

```text
x-axis: time
y-axis: hidden dimension
color: activation
```

Look for:

- Smooth temporal structure.
- Persistent patterns.
- Sudden instability.
- Dead dimensions.

## C. Stochastic uncertainty plot

Plot the average posterior standard deviation:

\[
\bar{\sigma}_t
=
\frac{1}{d_s}
\sum_j \sigma_{t,j}
\]

Compare it with events such as:

- Gripper contact.
- Cube occlusion.
- Pick-up.
- Drop.
- Collision.

## D. Multiple imagined futures

For one context and action sequence:

1. Keep the context fixed.
2. Sample different stochastic states.
3. Generate multiple rollouts.
4. Compare future branches.

This demonstrates what the stochastic component contributes.

***

# Phase 11: Final project structure

Use this structure:

```text
lecture4-rssm/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── deterministic_model.py
│   ├── stochastic_model.py
│   └── rssm.py
├── training/
│   ├── train_deterministic.py
│   ├── train_stochastic.py
│   └── train_rssm.py
├── evaluation/
│   ├── reconstruction.py
│   ├── rollout.py
│   ├── model_comparison.py
│   └── linear_probe.py
├── visualization/
│   ├── make_rollout_video.py
│   ├── plot_hidden_state.py
│   └── plot_uncertainty.py
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_deterministic_baseline.ipynb
│   ├── 03_stochastic_baseline.ipynb
│   ├── 04_rssm_states.ipynb
│   ├── 05_rssm_training.ipynb
│   └── 06_camera_off_dreaming.ipynb
├── checkpoints/
├── outputs/
└── README.md
```

# Final sequence

Follow this exact order:

```text
1. Run provided Lecture 4 code
2. Inspect SO101 trajectories
3. Reproduce deterministic baseline
4. Measure deterministic rollout failure
5. Implement stochastic baseline
6. Observe stochastic memory failure
7. Implement h_t transition
8. Implement prior p(s_t | h_t)
9. Implement posterior q(s_t | h_t, o_t)
10. Implement reparameterization
11. Train posterior reconstruction
12. Add KL loss
13. Implement observe_step
14. Implement imagine_step
15. Run camera-off dreaming
16. Compare deterministic, stochastic, and RSSM models
17. Reproduce linear probe and hidden-state analyses
18. Run multiple stochastic futures
19. Document results and failure cases
```

Your final understanding should be summarized in one sentence:

> The deterministic state \(h_t\) preserves temporal memory, the stochastic state \(s_t\) represents uncertain outcomes, the posterior learns from real observations, and the prior enables future prediction when observations are unavailable.