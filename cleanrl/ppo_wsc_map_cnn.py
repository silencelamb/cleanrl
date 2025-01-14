# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool
import logging

import gymnasium as gym
# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, "../")
from mapping_env import WSCMappingEnv

def get_root_logger(log_file, log_level=logging.DEBUG):
    """
    获取或创建名为 'ppo_wac_map_cnn' 的 logger。
    如果该 logger 不存在，它将被创建并配置。

    :param log_level: 日志级别
    :return: 配置好的 logger 对象
    """
    logger = logging.getLogger('ppo_wac_map_cnn')
    
    # 如果 logger 尚未配置
    if not logger.handlers:
        
        logger.setLevel(log_level)  # 设置日志级别
        
        # 创建一个handler，用于输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 给logger添加handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--gpuid", type=int, default=1,
        help="gpu id")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="pjlab-chip",
        help="the entity (team) of wandb's project")
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=25,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--hardware", type=str, choices=['gpu', 'wsc', 'dojo', 'wsgpu'], 
                        default="ws_gpu", help="hardware to use")
    parser.add_argument("--model-type", type=str, default="gpt",
        help="fixed model type, used in fix one graph mode")
    parser.add_argument("--model-size", type=str, default="350M",
        help="fixed model size, used in fix one graph mode")
    parser.add_argument("--constrain-mem",  action="store_true",
        help="whether to constrain memory usage")
    parser.add_argument("--use-offload", action="store_true",
                        help="use offload strategy (zero offoad -like)")
    parser.add_argument("--micro-batchsize", type=int,
                        help="micro batchsize")
    parser.add_argument("--rst-folder", type=str, default="")
    args = parser.parse_args()
    # when use use offload strategy, the memory contraint is off
    if args.use_offload:
        args.constrain_mem = False
    if args.hardware == 'dojo':
        args.num_steps = 25
    elif args.hardware == 'wsgpu':
        args.num_steps = 24
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(seed, **kwargs):
    def thunk():
        
        env = WSCMappingEnv(**kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        channel = envs.single_observation_space.shape[0]
        features_dim = 512
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(channel, 256, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(envs.single_observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(layer_init(nn.Linear(n_flatten, features_dim)), nn.ReLU())
        self.critic = nn.Sequential(
            self.cnn,
            self.linear,
            layer_init(nn.Linear(features_dim, 1), std=1.0),
        )
        num_actions = envs.single_action_space.nvec
        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                self.cnn,
                self.linear,
                layer_init(nn.Linear(features_dim, n), std=0.01),
            ) for n in num_actions
        ])

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_masks, action=None):
        logits_list = [actor_head(x) for actor_head in self.actor_heads]
        logits_0 = logits_list[0]
        probs_0 = Categorical(logits=logits_0)
        
        logits_1 = logits_list[1]
        action_masks = action_masks.type(torch.BoolTensor).to(device)
        logits_1 = torch.where(action_masks, logits_1, torch.tensor(-1e+9).to(device))
        probs_1 = Categorical(logits=logits_1)
        probs_list = [probs_0, probs_1]
        if action is None:
            action = torch.stack([probs.sample() for probs in probs_list], dim=-1)
        log_probs = torch.stack([probs.log_prob(act) for probs, act in zip(probs_list, torch.unbind(action, dim=-1))], dim=-1)
        total_log_prob = log_probs.sum(dim=-1)
        entropy = sum([probs.entropy() for probs in probs_list])
        
        return action, total_log_prob, entropy, self.critic(x)


if __name__ == "__main__":
    global_config = {}
    args = parse_args()
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.rst_folder = f"{args.rst_folder}/{args.hardware}-{args.model_type}-{args.model_size}-{args.micro_batchsize}-{date_str}"
    os.makedirs(args.rst_folder, exist_ok=True)
    logger = get_root_logger(log_file=f"{args.rst_folder}/log.txt")
    logger.info(f'args: {args}')

    run_name = f"{args.exp_name}_{args.hardware}_{args.model_type}{args.model_size}_micro{args.micro_batchsize}"+ \
        f"_mem{args.constrain_mem}_offload{args.use_offload}_{args.seed}_{date_str}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i,
                  use_image=True,
                  hardware=args.hardware,
                  offload=args.use_offload,
                  constrain_mem=args.constrain_mem,
                  logger=logger,
                  ) \
                      for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    for i in range(args.num_envs):
        # set the model type and size, so that the env can get the correct reward
        print(f"Using {args.model_type} {args.model_size} {args.micro_batchsize}")
        envs.envs[i].env.set_model_type_size(args.model_type, args.model_size, args.micro_batchsize)
        
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    num_action_dimensions = len(envs.single_action_space.nvec)
    actions = torch.zeros((args.num_steps, args.num_envs, num_action_dimensions)).to(device)
    action_masks_all = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.nvec[-1])).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    exception_caught = False  # Flag to indicate if an exception was caught

    for update in range(1, num_updates + 1):
        if exception_caught:
            # Reset the environment and start over
            next_obs = torch.Tensor(envs.reset()[0]).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
            exception_caught = False  # Reset the flag
            
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            action_masks = [env.env.get_action_mask() for env in envs.envs]
            action_masks = torch.Tensor(np.array(action_masks))
            action_masks = action_masks.type(torch.BoolTensor).to(device)
            
            # ALGO LOGIC: action logic
            with torch.no_grad():

                action, logprob, _, value = agent.get_action_and_value(
                                                next_obs, action_masks)
                values[step] = value.flatten()
            actions[step] = action
            action_masks_all[step] = action_masks
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            try:
                next_obs, reward, done, truncted, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
            except Exception as e:
                print(f"Exception occurred: {e}")
                exception_caught = True
                break  # Break out of the steps loop
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
        
        if exception_caught:
            continue  # Skip the rest of the code in this update and start over

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_actions_masks = action_masks_all.reshape((-1,envs.single_action_space.nvec[-1]))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                              b_actions_masks[mb_inds],
                                                                              b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
