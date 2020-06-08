# Optimistic Constrained Reinforcement Learning

This repository implements the algorithms presented in the [paper]()

# Dependencies

* We advise the reader to use virtualenv so that installing dependencies is easy

## Installation

```bash
python -m pip install -e .
```

# Code Arguments 

```bash
> python -u run.py --help
usage: run.py [-h] [--map MAP] [--alg {baseline,optimistic,appropo}]
              [--rounds ROUNDS] [--seed SEED]
              [--solver {ucbvi,policy_iteration,reinforce,value_iteration,a2c}]
              [--horizon HORIZON] [--output_dir OUTPUT_DIR]
              [--env {box_gridworld,gridworl}] [--budget BUDGET [BUDGET ...]]
              [--randomness RANDOMNESS] [--value_coef VALUE_COEF]
              [--entropy_coef ENTROPY_COEF]
              [--actor_critic_lr ACTOR_CRITIC_LR]
              [--discount_factor DISCOUNT_FACTOR]
              [--num_episodes NUM_EPISODES]
              [--conplanner_iter CONPLANNER_ITER] [--bonus_coef BONUS_COEF]
              [--planner {value_iteration,a2c}]
              [--optimistic_lambda_lr OPTIMISTIC_LAMBDA_LR]
              [--optomistic_reset {warm-start,scratch,continue}]
              [--num_fic_episodes NUM_FIC_EPISODES] [--mx_size MX_SIZE]
              [--proj_lr PROJ_LR] [--baseline_lambda_lr BASELINE_LAMBDA_LR]
```

# Running the code

To run the experiments, go to the directory `ocrl/`,

for the different environments:
  * Gridworld use the flag `--env gridworld`
  * Box Gridworld use the flag `--env box_gridworld`
  
for different instantions of our algorithm:
  * Value-Iteration planner use the flag `--planner value_iteration`
  * Actor-Critic planner use the flag `--planner a2c`
  
Commands to reproduce results in our paper:
  * Gridworld + Value-Iteration, run `python -u run.py --alg optimistic --env gridworld --planner value_iteration`
  * Gridworld + Actor-Critic, run `python -u run.py --alg optimistic --env box_gridworld --planner a2c`
  * Box Gridworld + Value-Iteration, run `python -u run.py --alg optimistic --env box_gridworld --planner value_iteration`
  * Box Gridworld + Actor-Critic, run `python -u run.py --alg optimistic --env box_gridworld --planner a2c`
  
The results are generated and stored in the location specified by `--output_dir` folder
