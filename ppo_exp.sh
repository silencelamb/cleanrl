# origin ppo
python cleanrl/ppo.py --exp-name ppo_origin --track --seed 1 --env-id CartPole-v0  --total-timesteps 50000

# cut rect, add action mask
python cleanrl/ppo_cutrect.py --exp-name action_mask --track --seed 1  --env-id  cut_rect --total-timesteps 150000