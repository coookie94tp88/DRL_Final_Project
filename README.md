# DRL Final Project

Course final project repo — **one folder per student** for side-by-side review.

| Folder | Author | Contents |
| --- | --- | --- |
| [`b12901162/`](b12901162/) | **You (AJ)** | DooRL — HAPPO / MARL environment (`doorl/`), configs, tests, `TRAINING.md` |
| [`b13902055/`](b13902055/) | Teammate | OracleGambit-style env + SAC player training (`env.py`, `train_player.py`, …) |

## Run your code (AJ)

```bash
cd b12901162
pip install -r requirements.txt
python -m doorl.train --config config/default.yaml --run-name learn_4p_v2 --device cuda \
  --override env.num_players=4 --override env.max_rounds=50 \
  --override train.total_timesteps=2000000
```

See [`b12901162/TRAINING.md`](b12901162/TRAINING.md) for TensorBoard, checkpoints, `watch`, and eval.

## Run teammate code

```bash
cd b13902055
pip install -r requirements.txt
python train_player.py   # see their README.md
```

## Compare fairly

- Same repo, different folders — no mixed paths.
- Training runs and checkpoints stay under `b12901162/runs/` (gitignored).
- Do not commit large `.pt` files; use checkpoints locally or release separately if needed.
