<h1 align="center">Quasi-Optimal Experimental Design (QOED)</h1>

<p align="center">
  <strong>Learning What Matters: Adaptive Information-Theoretic Objectives for Robot Exploration</strong>
</p>

<p align="center">
  <a href="https://roboticsconference.org"><img src="https://img.shields.io/badge/RSS-2026-blue" alt="Paper"></a>
  <a href="https://www.youwei-yu.com/qoed"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Website"></a>
  <a href="https://arxiv.org/abs/2605.12084"><img src="https://img.shields.io/badge/arXiv-2605.12084-red" alt="arXiv Preprint"></a>
</p>

## Installation

Install the base package. You can run [Experiment 1](#experiment-1-quick-demos-of-jackal-and-franka) after this step.

```bash
conda create -n qoed python=3.12 -y
conda activate qoed
conda install conda-forge::uv -y
uv pip install -e . --no-cache
```

Install the [MJLab](https://github.com/mujocolab/mjlab) and [RWM](https://github.com/leggedrobotics/robotic_world_model) dependencies. You can run [Experiment 2](#experiment-2-model-based-policy-optimization) after this step.

```bash
uv pip install -e ".[mjlab]" --no-cache
```

## Roadmap

- [x] RSS experiments: Jackal, Franka, Go1-Quadruped
- [x] Code optimization, e.g., torch.compile() or JAX
- [x] Additional robot platforms, e.g., G1-Humanoid, Leap-Hand
- [ ] Additional demos, e.g., next-best-view or differentiable MPC
- [ ] All robots from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## Experiment 1. Quick Demos of Jackal and Franka

> [!IMPORTANT]
> Requires VRAM >= 1GB

```bash
qoed-jackal-demo --device cuda:0 --sweep
```

```bash
qoed-franka-demo --device cuda:0 --sweep
```

`--sweep` runs 10 seeds per method and prints the paper summary.

![QOED vs. QOED-Agnostic and BOED on the Jackal and Franka demos](boed/qoed_demo.gif)

## Experiment 2. Model-based Policy Optimization

> [!IMPORTANT]
> Requires VRAM >= 8GB

#### 2.1 Pretrain in Simulation (ETA: 1 hour)

Optional: train the dynamics model and policy. Pretrained checkpoints are provided under `logs/`. `--task` selects the robot: `go1`, `jackal`, `g1`, or `leap`.

```bash
python mbpo/rsl_rl/train.py --task go1 --mode pretrain --gpu-ids "[0, ]" --headless
```

Optional: play the pretrained policy in MuJoCo.

```bash
python mbpo/rsl_rl/play.py --task go1 --mode pretrain --gpu-ids "[0, ]" --viewer auto
```

#### 2.2 Online Learning

Run real-world online RL, using MuJoCo as the real-system proxy.

```bash
python mbpo/rsl_rl/train.py --task go1 --mode finetune --info-gain qoed --gpu-ids "[0, ]" --viewer auto
```

Available info-gain modes: `qoed`, `qoed-agnostic`, `boed`, and `nothing`.

<details name="robot" open>
<summary>Unitree Go1</summary>

![Go1 online learning with QOED, BOED, QOED-Agnostic, and no info gain](mbpo/envs/assets/snapshot/qoed_go1.gif)

</details>

<details name="robot">
<summary>Clearpath Jackal</summary>

![Jackal online learning with QOED, BOED, QOED-Agnostic, and no info gain](mbpo/envs/assets/snapshot/qoed_jackal.gif)

</details>

<details name="robot">
<summary>LEAP Hand</summary>

![LEAP Hand online learning with QOED, BOED, QOED-Agnostic, and no info gain](mbpo/envs/assets/snapshot/qoed_leap.gif)

</details>

## Miscell
Please consider cite our work if it is interesting.
```
@inproceedings{yu2026qoed,
    author    = {Youwei Yu and Jionghao Wang and Zhengming Yu and Wenping Wang and Lantao Liu},
    title     = {{Learning What Matters: Adaptive Information Theoretic Objectives for Robot Exploration}},
    booktitle = {Proceedings of Robotics: Science and Systems},
    year      = {2026},
    address   = {Sydney, Australia},
    month     = {July}
}
```
