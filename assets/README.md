Toy example figures used for discussion.

Files:
- `00_reward.png`
  - Reward/Q heatmap for the five scenario families.
  - The toy is a contextual bandit with a constant observation: `obs = 0` for all samples.
  - The reward is implemented as a sum of Gaussian peaks in action space, minus a small quadratic penalty:
    - `Q(a) = sum_k scale_k * exp(-||a - c_k||^2 / width_k) - 0.03 ||a||^2`
  - The red curve is the dataset support boundary. The dataset is sampled inside this support.

- `01_bimodal.png`
  - Uniform dataset over the full X-shaped support.
  - Two reward peaks at the upper-left and upper-right.

- `02_unimodal.png`
  - Same uniform-support dataset as the bimodal case.
  - The difference from `01_bimodal.png` is only the reward: the right peak is turned off, so only the upper-left mode is valuable.

- `03_trimodal.png`
  - Uniform-support dataset.
  - Three reward peaks: upper-left, upper-right, and bottom-center.

- `04_quadmodal.png`
  - Uniform-support dataset.
  - Four reward peaks: upper-left, upper-right, bottom-left, and bottom-right.

- `05_asym_data_(left>right)_q_(left<right).png`
  - Asymmetric dataset and asymmetric reward are intentionally mismatched.
  - Dataset density is higher on the left side than the right side.
  - Reward is lower on the left peak and higher on the right peak.
  - This case is useful for separating “follow the dataset” behavior from “follow the reward” behavior.

Panel format:
- `Data Ref Heatmap`
- `BC`
- `FQL-Q`
- `FQL-D`
- `TopK-1`
- `TopK-3`
- `sUOT-H(z=0.0)`
- `sUOT-H(z=0.1)`
