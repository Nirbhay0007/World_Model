# Initial Setup

Start by reproducing the provided Lecture 4 implementation before writing your own model.

## 1. Create a workspace

```bash
mkdir lecture4-rssm
cd lecture4-rssm

git clone https://github.com/RajatDandekar/build-a-world-model-from-scratch.git
```

Keep two folders:

```text
lecture4-rssm/
├── build-a-world-model-from-scratch/   # original resources
└── my-implementation/                 # your experiments
```

Do not modify the original repository.

## 2. Create the Python environment

```bash
conda create -n lecture4-rssm python=3.11
conda activate lecture4-rssm
```

Install the common packages:

```bash
pip install torch torchvision numpy matplotlib pandas tqdm jupyter opencv-python
```

Then enter the repository:

```bash
cd build-a-world-model-from-scratch
```

If the repository contains a requirements file, install it:

```bash
pip install -r requirements.txt
```

## 3. Locate Lecture 4 resources

From the repository root:

```bash
find . -maxdepth 4 -type f | sort
```

Search for relevant files:

```bash
find . -iname "*rssm*" \
       -o -iname "*robot*" \
       -o -iname "*so101*" \
       -o -iname "*lecture*"
```

Look for:

- Lecture 4 notebook.
- Lecture 4 Python files.
- Dataset instructions.
- Checkpoints.
- Colab notebook.
- README files.
- Slides or lecture notes.

The Lecture 4 video is about building an SO101 robotics world model using an RSSM based on the PlaNet/Dreamer approach. [youtube](https://www.youtube.com/watch?v=gNwczJjm-8o)

## 4. Run the original code

Start Jupyter:

```bash
jupyter notebook
```

Open the Lecture 4 notebook and run it from top to bottom.

Do not change code yet.

Record:

```text
Python version:
PyTorch version:
Dataset location:
Image shape:
Action shape:
Hidden-state dimension:
Stochastic-state dimension:
Batch size:
Sequence length:
Learning rate:
Number of epochs:
```

Create a file:

```text
lecture4_notes.md
```

## 5. Check your hardware

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check the PyTorch version:

```bash
python -c "import torch; print(torch.__version__)"
```

If CUDA is available, use the GPU. Otherwise, start with a small dataset and short sequences.

## 6. Inspect the dataset

Before training, load one episode and verify:

```text
image shape
action shape
image value range
action value range
number of frames
```

You are looking for data conceptually like:

```text
images:  [time, 3, 64, 64]
actions: [time, 6]
```

Display several frames in order:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for t, ax in enumerate(axes):
    ax.imshow(images[t])
    ax.set_title(f"t={t}")
    ax.axis("off")

plt.show()
```

Plot the six action dimensions:

```python
plt.figure(figsize=(12, 4))

for joint in range(6):
    plt.plot(actions[:, joint], label=f"action {joint}")

plt.legend()
plt.xlabel("time")
plt.show()
```

## 7. Verify the transition

Confirm that the data follows:

```text
image[t] + action[t] → image[t+1]
```

Check two consecutive frames and the corresponding action. This prevents a time-alignment error that can make the model impossible to train.

## 8. Create your experiment folder

Inside `my-implementation`, create:

```text
my-implementation/
├── notebooks/
│   └── 01_dataset_inspection.ipynb
├── models/
├── training/
├── evaluation/
├── checkpoints/
└── outputs/
```

Copy only the files you need from the original repository. Keep the original code untouched for comparison.

## 9. Your first milestone

Do not start with RSSM.

Your first milestone is:

```text
Environment/dataset loads
        ↓
One episode is visualized
        ↓
Image and action shapes are verified
        ↓
Original Lecture 4 code runs
        ↓
Results are saved
```

After this, begin with the deterministic baseline:

```text
CNN encoder → GRU → decoder
```

Then continue in this order:

```text
1. Dataset inspection
2. Deterministic baseline
3. Pure stochastic model
4. RSSM prior and posterior
5. Reconstruction and KL losses
6. Observe mode
7. Imagine mode
8. Camera-off rollout
```

### Your immediate task

For now, do only these three things:

1. Clone the repository.
2. Run the provided Lecture 4 notebook/code.
3. Record the dataset and tensor shapes in `lecture4_notes.md`.

Do not implement the RSSM until these initial steps work.