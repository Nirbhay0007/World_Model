Here is a step-by-step roadmap to build and simulate **Mini Flappy Bird** following the exact world model architecture from Lecture 3 (Ha & Schmidhuber style: VAE + Recurrent Dynamics).

---

### Phase 1: Environment Setup & Data Collection

1. **Build a Lightweight Engine:**
* Implement a minimalist Flappy Bird environment in Python (using `pygame` or headless `gymnasium`/NumPy).
* **Grid:** Render frames in low resolution, either $32 \times 32$ or $64 \times 64$ pixels with 3 RGB channels (around 3,000 to 12,000 numbers per frame).


* **Bird:** Render as a high-contrast, bright block (e.g., 2 to 4 bright yellow pixels) to make tracking manageable.


* **Pipes:** Green vertical bars with a fixed-height gap that scroll left by 1 pixel per step.


* **Actions:** Define 2 discrete actions: $a_t \in \{0, 1\}$ (`0 = Do Nothing / Fall`, `1 = Flap / Jump`).




2. **Collect Unsupervised Rollouts:**
* Use a **random policy** to explore the physics: Flap with probability $p \approx 0.15$ to $0.20$ per step.


* Collect roughly **200 episodes** of **100–120 steps** each (giving ~20,000 to 24,000 paired tuples: frame $o_t$ and action $a_t$).


* The physics stays the same whether the player crashes or survives; you do not need an expert player. Save the collected dataset to disk as NumPy arrays.





---

### Phase 2: Vision Model (Variational Autoencoder)

1. **Architecture:**
* **Encoder:** 3 to 4 Convolutional layers (with stride 2) compressing the image down to a compact continuous latent vector $z_t \in \mathbb{R}^{16}$ or $\mathbb{R}^{32}$ (predicting mean $\mu$ and log-variance $\log \sigma^2$).


* **Decoder:** Transposed convolutions mapping $z_t$ back to the original image dimensions.




2. **Crucial Loss Fix (From Mini Pong):**
* A standard MSE reconstruction loss will cause the network to reconstruct the large green pipes accurately while **completely erasing the small bird**, because 4 bird pixels out of 1,024 contribute negligible error.


* **Fix:** Apply a weighted MSE loss that heavily penalizes errors on bright/foreground bird pixels relative to empty background pixels.


* Train the VAE until reconstructions preserve sharp pipe edges and clear bird visibility. Once trained, freeze the VAE and pre-encode all 200 episodes into latent sequences $\{z_0, z_1, \dots, z_T\}$.





---

### Phase 3: Dynamics / Memory Model (RNN / GRU)

1. **Architecture:**
* Single-frame observations lack velocity information (a static snapshot cannot reveal whether the bird is ascending or falling under gravity).


* Build a **Recurrent Network (GRU or LSTM)** with a hidden state dimension of $h \in \mathbb{R}^{128}$ or $\mathbb{R}^{256}$.


* **Input at step $t$:** Concatenation of current latent and action $[z_t, a_t]$ along with previous hidden state $h_t$.


* **Memory Update:** $h_{t+1} = \text{GRU}([z_t, a_t], h_t)$.


* **Prediction Head:** A linear/MLP head on top of $h_{t+1}$ that predicts the next latent vector $\hat{z}_{t+1}$.




2. **Training:**
* Train over trajectory slices minimizing the MSE loss:

$$\mathcal{L}_{\text{dynamics}} = \Vert{}\hat{z}_{t+1} - z_{t+1}^{\text{target}}\Vert{}^2$$



between the predicted code and the actual VAE-encoded code from the next frame in the dataset.


* Track that loss decreases across time steps as the recurrent memory fills up with velocity history.





---

### Phase 4: Interactive Simulation ("Dreaming" with Engine Off)

1. **Closed-Loop Unrolling:**
* Initialize memory $h_0$ as a zero vector and feed in the initial starting frame $o_0$ encoded into $z_0$.


* **Turn the game engine off entirely:**

1. Player selects action $a_t$ (e.g., via keyboard spacebar).


2. Update recurrent memory: $h_{t+1} = \text{GRU}([z_t, a_t], h_t)$.


3. Predict next latent code: $\hat{z}_{t+1} = \text{MLP}(h_{t+1})$.


4. Decode into a visual image: $\hat{o}_{t+1} = \text{Decoder}(\hat{z}_{t+1})$.


5. Feed $\hat{z}_{t+1}$ back as the input latent for step $t+1$.






2. **Validation & Next Steps:**
* Verify whether the bird bounces upward when flapping and smoothly falls under simulated gravity.


* Check for **compounding error**: In purely deterministic models, autoregressive feedback can cause scrolling pipe edges to blur or warp after 20–40 steps. If pipe blur occurs, the natural progression is to upgrade from Lecture 3's deterministic GRU to Lecture 4's **RSSM (combining deterministic state $h$ with stochastic state $s$)**.





Would you like the PyTorch skeleton code for either the weighted VAE or the GRU dynamics predictor to get started?