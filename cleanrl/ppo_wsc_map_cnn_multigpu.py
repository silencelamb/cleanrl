# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
import warnings
from distutils.util import strtobool

import gymnasium as gym
# import gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, "../")
from mapping_env import WSCMappingEnv



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=30,
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
    parser.add_argument("--device-ids", nargs="+", default=[],
        help="the device ids that subprocess workers will use")
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl", "mpi"],
        help="the id of the environment")
    parser.add_argument("--model-type", type=str, default="gpt",
        help="fixed model type, used in fix one graph mode")
    parser.add_argument("--model-size", type=str, default="350M",
        help="fixed model size, used in fix one graph mode")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        
        env = WSCMappingEnv(use_image=True)
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
        # if local_rank == 0:
        #     print(f"logits_0: {logits_0}")
        
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
    # torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py
    # taken from https://pytorch.org/docs/stable/elastic/run.html
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    args = parse_args()
    args.world_size = world_size
    args.num_envs = int(args.num_envs / world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    if world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=world_size)
    else:
        warnings.warn(
            """
Not using distributed mode!
If you want to use distributed mode, please execute this script with 'torchrun'.
E.g., `torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py`
        """
        )
    print(f"================================")
    print(f"args.num_envs: {args.num_envs}, args.batch_size: {args.batch_size}, args.minibatch_size: {args.minibatch_size}")
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = None
    if local_rank == 0:
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
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if len(args.device_ids) > 0:
        assert len(args.device_ids) == world_size, "you must specify the same number of device ids as `--nproc_per_node`"
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device_count = torch.cuda.device_count()
        if device_count < world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")


    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    for i in range(args.num_envs):
        # set the model type and size, so that the env can get the correct reward
        if local_rank == 0:
            print(f"Using {args.model_type} {args.model_size}")
        envs.envs[i].env.set_model_type_size(args.model_type, args.model_size)
        
    agent = Agent(envs).to(device)
    torch.manual_seed(args.seed)
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
    num_updates = args.total_timesteps // (args.batch_size * world_size)
    
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
            global_step += 1 * args.num_envs * world_size
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

            if local_rank == 0 and "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

        if exception_caught:
            continue  # Skip the rest of the code in this update and start over

        print(f"local_rank: {local_rank}, action.sum(): {action.sum()}, update: {update}")
        print(f"agent.actor_heads[0][0][0].weight.sum(): {agent.actor_heads[0][0][0].weight.sum()}")
        print(f"agent.actor_heads[0][1][0].weight.sum(): {agent.actor_heads[0][1][0].weight.sum()}")
        print(f"agent.actor_heads[1][0][0].weight.sum(): {agent.actor_heads[1][0][0].weight.sum()}")
        print(f"agent.actor_heads[1][1][0].weight.sum(): {agent.actor_heads[1][1][0].weight.sum()}")
                    
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
                if world_size > 1:
                    # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in agent.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in agent.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
                            )
                            offset += param.numel()
                            
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if local_rank == 0:        
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
    if local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()