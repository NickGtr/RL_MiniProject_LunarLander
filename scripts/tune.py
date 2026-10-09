"""Recherche des meilleurs hyperparamètres de DDPG ou TD3 avec Optuna

    uv run python scripts/tune.py --algo td3 --n_trials 20 --max_steps 100000 --seeds 1 2

Optuna choisit les valeurs de chaque essai dans les intervalles de `suggest_config`. 
Le score d'un essai est la moyenne, sur les seeds, des 3 dernières évaluations .

L'étude est sauvegardée dans `outputs/optuna/<algo>.db`,
les logs TensorBoard de chaque essai dans `outputs/optuna/<algo>/trial_<n>/seed_<s>/`.

Voir les résultats :
    uvx optuna-dashboard sqlite:///outputs/optuna/td3.db
"""

import argparse
from pathlib import Path

import optuna

from rl_miniproject_lunarlander import DDPGConfig, TD3Config, run_ddpg, run_td3

OPTUNA_DIR = Path("outputs/optuna")

def suggest_config(trial: optuna.Trial, algo: str, max_steps: int, seed: int):
    """L'espace de recherche : les intervalles dans lesquels Optuna choisit les hyperparamètres"""
    # Optuna n'accepte pas de tuples comme choix : on lui donne du texte, "64,64" -> (64, 64)
    hidden = tuple(int(n) for n in trial.suggest_categorical("hidden", ["64,64", "256,256"]).split(","))
    params = dict(
        max_steps=max_steps,
        seed=seed,
        gamma=trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        tau=trial.suggest_float("tau", 1e-3, 0.05, log=True),
        lr_actor=trial.suggest_float("lr_actor", 1e-4, 1e-3, log=True),
        lr_critic=trial.suggest_float("lr_critic", 1e-4, 1e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        action_noise=trial.suggest_float("action_noise", 0.05, 0.3),
        actor_hidden=hidden,
        critic_hidden=hidden,
        eval_interval=5_000,
    )
    if algo == "ddpg":
        return DDPGConfig(**params)
    # Si c'est TD3, on ajoute policy_delay et target_noise
    return TD3Config(
        **params,
        policy_delay=trial.suggest_int("policy_delay", 1, 3),
        target_noise=trial.suggest_float("target_noise", 0.1, 0.3),
    )


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
# Sans --algo, argparse arrête le script
parser.add_argument("--algo", choices=["ddpg", "td3"], required=True)
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=10_000)
parser.add_argument("--seeds", type=int, nargs="+", default=[1])
args = parser.parse_args()

run = run_ddpg if args.algo == "ddpg" else run_td3


def objective(trial: optuna.Trial) -> float:
    scores = []
    for seed in args.seeds:
        cfg = suggest_config(trial, args.algo, args.max_steps, seed)
        run_dir = OPTUNA_DIR / args.algo / f"trial_{trial.number}" / f"seed_{seed}"
        evaluator = run(cfg, run_dir=run_dir)
        evaluator.writer.close()
        scores.append(sum(r.mean for r in evaluator.history[-3:]) / len(evaluator.history[-3:]))
    return sum(scores) / len(scores)


OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
study = optuna.create_study(
    study_name=args.algo,
    storage=f"sqlite:///{OPTUNA_DIR / args.algo}.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.n_trials)

print(f"Meilleur essai : n°{study.best_trial.number}, score {study.best_value:.1f}")
print(study.best_params)
