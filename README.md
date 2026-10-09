# RL_MiniProject_LunarLander

DDPG et TD3 sur `LunarLanderContinuous-v3` (Gymnasium), avec la bibliothèque du cours `rl_mind`.

## Organisation

```
src/rl_miniproject_lunarlander/
    algos/          run_ddpg, run_td3 et leurs configs (DDPGConfig, TD3Config)
    actors/         acteur déterministe continu + bruit d'exploration
    critics/        réseau Q continu
    losses/         pertes de l'acteur et du critique
scripts/tune.py     recherche d'hyperparamètres avec Optuna
use_example.ipynb   exemple d'entraînement et de visualisation dans un notebook
outputs/            résultats des entraînements
```

## Chercher les meilleurs hyperparamètres : `scripts/tune.py`

Un lancement optimise **un seul algorithme** (`--algo ddpg` ou `--algo td3`, pas les deux dans une même commande).
Exemple :
```
uv run python scripts/tune.py --algo td3 --n_trials 20 --max_steps 10000 --seeds 1 2
```

- `--n_trials` : nombre d'essais (20 par défaut)
- `--max_steps` : pas d'entraînement par essai (10 000 par défaut)
- `--seeds` : seeds entraînées pour chaque essai (`1` par défaut)

Optuna choisit les hyperparamètres de chaque essai en s'appuyant sur les essais
précédents. Les intervalles de recherche peuvent être choisis dans `suggest_config`.
Le score d'un essai est la moyenne des 3 dernières évaluations pour chaque seed.

### Résultats

Tout est rangé dans `outputs/optuna/` : l'étude dans `<algo>.db`, les logs TensorBoard
de chaque entraînement dans `<algo>/trial_<n>/seed_<s>/`.

- Les meilleurs hyperparamètres sont affichés à la fin, ou dans le tableau de bord :
  `uvx optuna-dashboard sqlite:///outputs/optuna/td3.db`
- Les courbes d'apprentissage de chaque essai sont accessibles via TensorBoard :
  `uv run tensorboard --logdir outputs/optuna/td3`
