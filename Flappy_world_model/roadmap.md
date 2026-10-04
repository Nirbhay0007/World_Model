# Mini Flappy Bird: World Model Architecture & Neural Dreaming

An end-to-end PyTorch implementation of a **World Model** (based on Ha & Schmidhuber, 2018) applied to a minimalist custom Flappy Bird environment (`MiniFlappy`). The model learns to compress visual frames into compact latent vectors, model physical dynamics over time, and simulate full interactive gameplay entirely inside its own imagination with the game engine turned off.

---

## 🏗️ Architecture Overview

The system consists of three foundational components:

1. **Environment & Data Engine (`MiniFlappy`):**
   - 32x32 RGB canvas (3,072 values per frame).
   - Physics: Constant gravity (+0.3 px/step), flap impulse (-1.2 px/step), terminal speed (3.0 px/step), scrolling green pipes with vertical gaps.
   - Discrete action space: `0 = Fall (Gravity)`, `1 = Flap (Impulse)`.

2. **Vision Model (V - Variational Autoencoder):**
   - **Encoder:** 3-layer Convolutional network (Conv2D -> BatchNorm -> ReLU) with stride 2, projecting high-dimensional frames down to a 32-dimensional continuous latent space (`z_dim = 32`).
   - **Decoder:** 3-layer Transposed Convolutional network (ConvTranspose2D -> BatchNorm -> ReLU -> Sigmoid) mapping `z` back to the 32x32 RGB reconstruction.
   - **Custom Weighted Reconstruction Loss:** Implemented a 15x pixel-weighting mask on bright bird and pipe pixels to prevent standard MSE from ignoring the tiny 2x2 bird footprint.

3. **Memory / Dynamics Model (M - Recurrent GRU):**
   - Single-layer GRU (`hidden_dim = 128`, `input_dim = 33` = 32 latent + 1 action).
   - Linear prediction head mapping hidden state `h_(t+1)` to predicted next latent state `z_hat_(t+1)`.
   - Loss: Latent Mean Squared Error (MSE) between predicted latent vector and target VAE-encoded latent vector.

4. **Closed-Loop Simulator (Dreaming):**
   - Game engine is completely disconnected.
   - The model takes a single seed frame `z_0`, receives action sequences, and autoregressively feeds its own predicted `z_hat_t` back as input `z_(t+1)` to hallucinate long-horizon trajectories.

---

## 🚀 Completed Milestones & Results

### Phase 1: Environment Setup & Data Collection
- Generated **200 unsupervised exploration episodes** of **100 steps each** (20,000 total frames).
- Stored as PyTorch tensors:
  - Frames: `(20000, 3, 32, 32)` normalized float32 tensor.
  - Actions: `(200, 100, 1)` binary tensor.

### Phase 2: Vision Model Training & Pre-Encoding
- Trained the VAE for 15 epochs using Adam optimizer (`lr = 1e-3`).
- Compressed all 20,000 frames from **234.4 MB** down to an ultra-compact **2.44 MB** `latent_dataset` tensor of shape `(200, 100, 32)` (99% data compression ratio).
- Reconstructions accurately preserve sharp green pipe boundaries and clear yellow bird positions.

### Phase 3: Dynamics Model Training & 1-Step Validation
- Trained the GRU over 20 epochs across batch size 32.
- **Training Convergence:** Latent MSE loss dropped by **~90%** (from `0.3101` down to `0.0325`).
- **1-Step Visual Verification:** Side-by-side comparison between ground truth frames and decoded 1-step predictions demonstrated accurate physics tracking for falling, jumping, and pipe motion.

### Phase 4: Closed-Loop Dreaming (Neural Simulation)
- **Gravity Fall Experiment (`[0] * 30`):**
  - Model imagined quadratic downward acceleration, reached the floor in exactly 10 steps, maintained floor boundary collision, and scrolled the pipe across the screen.
- **Periodic Flapping Flight (`[1, 0, 0, 0, 0, ...]`):**
  - When given periodic flap commands (every 5 steps), the model hallucinated upward lift against gravity and navigated the bird directly through the scrolling green pipe gap over a 30-step filmstrip.
- **Warm-Start vs. Cold-Start Dynamics:**
  - Discovered that providing a 5-step warm-up memory (`h_5`) provides crucial velocity context that completely eliminates initial cold-start ghosting artifacts.

---

## 📂 Project Structure

```text
World_Model/
├── Flappy_world_model/
│   ├── mini_floppy.ipynb   # Main Jupyter notebook containing complete implementation
│   ├── roadmap.md          # Project documentation & milestone report
│   ├── vae_model.pth       # Saved weights for trained VAE Vision Model
│   └── memory_model.pth    # Saved weights for trained GRU Dynamics Model
└── .agents/                # Coding guidelines & agent rules
```

---

## 🛠️ How to Run & Reproduce

1. **Prerequisites:**
   ```bash
   pip install torch torchvision numpy matplotlib ipywidgets
   ```

2. **Open the Notebook:**
   Launch Jupyter and open `Flappy_world_model/mini_floppy.ipynb`.

3. **Execution Flow:**
   - **Cells 1–10:** Environment creation & rollout collection.
   - **Cells 11–20:** VAE definition, training, and dataset pre-encoding.
   - **Cells 21–35:** GRU Memory Model definition, training, and 1-step validation.
   - **Cells 36–43:** Closed-loop dream simulation, filmstrips, and model checkpoint saving.

---

## 🔬 Core Insights & Key Learnings

1. **Pixel Space vs. Latent Space Dynamics:**
   Predicting future states in a 32-dimensional latent space is orders of magnitude faster and computationally lighter than predicting raw 3,072-pixel matrices directly.

2. **Weighted Reconstruction Loss:**
   Without spatial loss weighting, small critical objects (like a 4-pixel bird on a 1,024-pixel grid) get averaged out as background noise by standard MSE loss.

3. **Compounding Autoregressive Error:**
   Pure deterministic world models can sustain realistic imagination horizons for 15–25 steps. To extend imagination horizons further without blur or mode collapse, advanced architectures introduce stochastic latent transitions (e.g. RSSM in Dreamer).