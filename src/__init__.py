"""Deterministic scenario mining as an auditable reference for VLM auto labelling.

Module map (see PLAN.md for the stage each one belongs to, ARCHITECTURE.md for how they
fit together):
    data.py         nuScenes loading, paths, split helpers, inference bundles
    groundtruth.py  the deterministic labeller: kinematics, manoeuvres, map, objects
    bev.py          bird's-eye-view rendering, and the symbolic scene description
    vlm.py          model backends (MLX locally / Transformers on a T4)
    prompts.py      every prompt, versioned, in one place
    eval.py         per-tag precision / recall / F1, baselines, bootstrap, significance
    figures.py      every rendered figure
    storyboard.py   scene to timed event sequence, and its evaluation
    scenario.py     structured scenario description, and the OpenSCENARIO export
    sim.py          MetaDrive / ScenarioNet instantiation check (runs in ./venv-sim)
    bdd.py          BDD100K signage subset and the cross-dataset comparison
    signage.py      traffic-light and sign presence from the nuScenes map
    nuqa.py         external validation against NuScenes-QA

`vlm.py` and `prompts.py` are deliberately free of numpy, so both model environments can
import them: mlx-vlm requires numpy>=2 and the nuScenes devkit breaks on it.
"""
