from abc import ABC, abstractmethod
from hashlib import sha256
import json
import logging
import time
import pandas as pd
from typing import Dict
from collections import OrderedDict, deque
import numpy as np


def get_custom_logger(name="custom_logger"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "(%(process)d) [%(levelname).1s] - (%(asctime)s) >> %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate output via root logger
    return logger

#create a logger using custom logger function
logging = get_custom_logger("custom_logger")

class Logger(ABC):
    #def __init__(self, cfg: DictConfig):
    #    self.config = OmegaConf.to_container(cfg)

    def __init__(self, cfg: Dict):
        self.config = cfg

        non_hash_keys = ["seed"]
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in self.config.items() if k not in non_hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self.database = None

        self.last_time = time.time()
        self.last_update = 0
        self.last_step = 0

        # Rolling episode tracking (last 10 episodes)
        self._episode_returns = deque(maxlen=10)
        self._episode_mols_found = deque(maxlen=10)

        # Action type statistics (rolling window of recent actions)
        self._action_stats = {'valid': 0, 'invalid': 0, 'noop': 0, 'total': 0}

        # Loss history for trend tracking
        self._loss_history = {}
    
    def update_action_stats(self, infos_list):
        """Update action type statistics from step-level infos.
        
        Args:
            infos_list: List of info dicts from env.step(), each containing 'is_noop' and 'is_invalid'
        """
        for infos in infos_list:
            for info in infos:
                self._action_stats['total'] += 1
                if info.get('is_noop', False):
                    self._action_stats['noop'] += 1
                elif info.get('is_invalid', False):
                    self._action_stats['invalid'] += 1
                else:
                    self._action_stats['valid'] += 1
    
    def get_action_stats_summary(self, reset=False):
        """Get action type percentages and optionally reset counters.
        
        Returns:
            dict with 'valid_pct', 'invalid_pct', 'noop_pct', 'total'
        """
        total = self._action_stats['total']
        if total == 0:
            return {'valid_pct': 0.0, 'invalid_pct': 0.0, 'noop_pct': 0.0, 'total': 0}
        
        summary = {
            'valid_pct': 100 * self._action_stats['valid'] / total,
            'invalid_pct': 100 * self._action_stats['invalid'] / total,
            'noop_pct': 100 * self._action_stats['noop'] / total,
            'total': total
        }
        
        if reset:
            self._action_stats = {'valid': 0, 'invalid': 0, 'noop': 0, 'total': 0}
        
        return summary
    
    def log_action_stats(self, reset=True):
        """Log action type statistics and optionally reset counters."""
        stats = self.get_action_stats_summary(reset=reset)
        if stats['total'] > 0:
            self.info(f"  Action types: Valid {stats['valid_pct']:.1f}% | Invalid {stats['invalid_pct']:.1f}% | NoOp {stats['noop_pct']:.1f}%  (n={stats['total']})")
        

    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        logging.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        logging.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        logging.critical(*args, **kwargs)
    
    def watch(self, model):
        self.debug(model)
    
    @abstractmethod
    def log_metrics(self, d: Dict, step_name: str, step: int):
        ...

    @abstractmethod
    def log_eval(self, d: Dict, step_name: str, step: int):
        ...
    
    def completed_run(self):
        pass

    def failed_run(self):
        pass
    
    def log_progress(
        self, infos, update, step, total_steps, loss_dict=None, label="Train"
    ):
        elapsed = time.time() - self.last_time
        self.last_time = time.time()
        ups = (update - self.last_update) / max(elapsed, 1e-6)
        self.last_update = update
        steps_elapsed = step
        fps = (steps_elapsed - self.last_step) / max(elapsed, 1e-6)
        self.last_step = step

        self.info(f"Updates {update}, Environment timesteps {steps_elapsed}")
        self.info(
            f"UPS: {ups:.1f}, FPS: {fps:.1f}, {steps_elapsed}/{total_steps} ({100 * steps_elapsed/total_steps:.2f}%) completed"
        )

        # ── Per-agent losses ──────────────────────────────────────────
        if loss_dict is not None:
            n_agents = len([k for k in loss_dict if k.startswith("agent_") and k.endswith("/entropy")])
            header = f"{'Agent':>7} | {'Value Loss':>12} | {'Actor Loss':>12} | {'Entropy':>10}"
            self.info(header)
            self.info("-" * len(header))
            for i in range(1, n_agents + 1):
                v_loss = loss_dict.get(f'agent_{i}/value_loss', 0.0)
                a_loss = loss_dict.get(f'agent_{i}/actor_loss', 0.0)
                ent    = loss_dict.get(f'agent_{i}/entropy', 0.0)

                # Store history for trend detection
                for key, val in [(f'agent_{i}/value_loss', v_loss),
                                 (f'agent_{i}/actor_loss', a_loss),
                                 (f'agent_{i}/entropy', ent)]:
                    if key not in self._loss_history:
                        self._loss_history[key] = deque(maxlen=50)
                    self._loss_history[key].append(val)

                self.info(f"  Ag {i:>2}  | {v_loss:>12.5f} | {a_loss:>12.5f} | {ent:>10.5f}")

            # Entropy trend warning
            for i in range(1, n_agents + 1):
                ent_hist = self._loss_history.get(f'agent_{i}/entropy', [])
                if len(ent_hist) >= 5:
                    recent = list(ent_hist)[-5:]
                    if all(e < 0.5 for e in recent):
                        self.warning(f"Agent {i} entropy very low ({recent[-1]:.4f}) – policy may be collapsing")

        # ── Episode-level tracking ────────────────────────────────────
        if infos:
            for info in infos:
                ep_return = sum(info["all rewards"]) if isinstance(info["all rewards"], (list, tuple)) else float(info["all rewards"])
                ep_mols = info.get("unique_found", 0)
                self._episode_returns.append(ep_return)
                self._episode_mols_found.append(ep_mols)

            n_eps = len(self._episode_returns)
            avg_return = np.mean(self._episode_returns) if n_eps > 0 else 0.0
            std_return = np.std(self._episode_returns) if n_eps > 1 else 0.0
            avg_mols   = np.mean(self._episode_mols_found) if n_eps > 0 else 0.0
            total_mols  = sum(info.get("unique_found", 0) for info in infos)

            tag = "Train" if label == "Train" else "Eval"
            self.info(f"[{tag}] Last {n_eps} episodes:")
            self.info(f"  Avg return: {avg_return:.3f}  (std {std_return:.3f})")
            self.info(f"  Avg molecules found per episode: {avg_mols:.1f}")
            self.info(f"  Total molecules found this batch: {total_mols}")
        
        # Log action type statistics (accumulated since last log_progress call)
        self.log_action_stats(reset=True)
        self.info("-------------------------------------------")

    def log_episode(self, timestep, info, main_label="Train", loss_dict=None, print_train_log=True):
        
        #print('Logging episode within logger')
        #print(info)
        
        info["episode_reward"] = sum(info["episode_reward"])
        if "terminal_observation" in info:
            del(info["terminal_observation"])
        log_dict = {}
        for k, v in info.items():
            log_dict[f"{main_label}/{k.replace('/','_')}"] = v

        if main_label == "Train":
            self.log_metrics(log_dict, "timestep", timestep)
        else:
            self.log_eval(log_dict, "timestep", timestep)
        
        if main_label == "Train":
            if print_train_log:
                self.info(
                    f"Completed episode {info['completed_episodes']}: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
                )
        else:
            self.info(
                f"Completed evaluation: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
            )
    


class TensorboardLogger(Logger):
    def __init__(self, cfg, tensorboard_dir):
        super(TensorboardLogger, self).__init__(cfg)
        from torch.utils.tensorboard import SummaryWriter
        self.tensorboard_logger = SummaryWriter(tensorboard_dir)

    def log_metrics(self, d: Dict, step_name: str, step: int):
        for key, v in d.items():
            self.tensorboard_logger.add_text(key, v, step)

class FileSystemLogger(Logger):
    def __init__(self, cfg, project_name=None, print_train_log=True):
        super(FileSystemLogger, self).__init__(cfg)
        self.results_path = "results.csv"
        self.eval_path = "eval_results.csv"
        self.project_name = project_name
        self.print_train_log = print_train_log

    def log_metrics(self, d: Dict, step_name: str, step: int):
        keys_to_remove = [
            'Train/actions', 'Train/agent_id', 'Train/episode_time',
            'Train/infostr', 'Train/completed_episodes'
        ]
        for key in keys_to_remove:
            d.pop(key, None)
        ordered_dict = OrderedDict(sorted(d.items()))
        df = pd.DataFrame.from_dict([ordered_dict])
        with open(self.results_path, "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        #self.info(f"Metrics logged to {self.results_path} at {step_name}={step}")
        #for k, v in ordered_dict.items():
        
         #   self.info(f"\t{k} = {v}")

    def log_progress(
        self, infos, update, step, total_steps, loss_dict=None, label="Train"
    ):
        super().log_progress(infos, update, step, total_steps, loss_dict, label)

    def log_eval(self, d: Dict, step_name: str, step: int):
        keys_to_remove = [
            'Train/actions', 'Train/agent_id', 'Train/episode_time',
            'Train/infostr', 'Train/completed_episodes'
        ]
        for key in keys_to_remove:
            d.pop(key, None)
        ordered_dict = OrderedDict(sorted(d.items()))
        df = pd.DataFrame.from_dict([ordered_dict])
        with open(self.eval_path, "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        #self.info(f"Eval metrics logged to {self.eval_path} at {step_name}={step}")
        for k, v in ordered_dict.items():
            self.info(f"\t{k} = {v}")

    def log_episode(self, timestep, info, main_label="Train", print_train_log=True):
        
        #print('Logging episode within logger')
        #print(info)
        
        info["episode_reward"] = sum(info["episode_reward"])
        if "terminal_observation" in info:
            del(info["terminal_observation"])
        log_dict = {}
        for k, v in info.items():
            log_dict[f"{main_label}/{k.replace('/','_')}"] = v

        if main_label == "Train":
            self.log_metrics(log_dict, "timestep", timestep)
        else:
            self.log_eval(log_dict, "timestep", timestep)
        
        if main_label == "Train":
            if print_train_log:
                self.info(
                    f"Completed episode {info['completed_episodes']}: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
                )
        else:
            self.info(
                f"Completed evaluation: Steps = {info['episode_length']} / Total Return = {info['episode_reward']:.3f} / Total duration = {info['episode_time']}s"
            )
    
    

