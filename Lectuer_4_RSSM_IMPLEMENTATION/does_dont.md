## Do’s

- Run the provided Lecture 4 code before modifying it.
- Follow the implementation in this order: dataset → deterministic model → stochastic model → RSSM.
- Check image, action, \(h_t\), and \(s_t\) tensor shapes at every stage.
- Understand the difference between:
  - \(h_t\): deterministic memory.
  - \(s_t\): stochastic state.
- Train first with real observations, then test camera-off imagination.
- Track reconstruction loss and KL loss separately.
- Compare deterministic, stochastic, and RSSM rollouts.
- Save checkpoints, plots, and videos after every major step.
- Change only one thing at a time.

## Don’ts

- Don’t start directly with the complete RSSM.
- Don’t skip the deterministic and stochastic baselines.
- Don’t give the prior network access to future images.
- Don’t use real future frames during camera-off dreaming.
- Don’t confuse teacher-forced prediction with autonomous imagination.
- Don’t change the dataset, architecture, and hyperparameters simultaneously.
- Don’t judge the model only by training loss.
- Don’t scale up before a small example works.
- Don’t delete failed experiments—record what went wrong.