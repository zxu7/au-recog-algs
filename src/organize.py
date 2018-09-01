import os
import sys
import time
import json
import subprocess
from datetime import datetime


class WorkFlow(object):
    """setup Git, Config, ExperimentSaver, """
    def __init__(self, experiment_dir='experiments/'):
        self.new_experiment_dir = None
        self.config_path = None
        self.experiment_info_path = None

        # check experiment_dir exists
        assert os.path.isdir(experiment_dir), "experiment directory {} does not exist!".format(experiment_dir)
        self.experiment_dir = experiment_dir
        # init Git
        self.git = Git()
        self.config = None
        pass

    def start(self, config):
        """create experiment folder, save git info & script, config, etc
        :param config: BaseConfig object or its child;
        :return: None
        """
        self.config = config

        # create experiment folder, initialize config, exp_info
        print("creating new experiment folder...")
        self._create_new_experiment_dir()
        self.config_path = os.path.join(self.new_experiment_dir, 'experiment.config')
        self.experiment_info_path = os.path.join(self.new_experiment_dir, 'info.txt')

        # save config
        print("saving experiment config...")
        self._write_config(self.config_path)

        # save experiment info
        print("saving experiment info...")
        self._write_experiment_info(self.experiment_info_path)

        return None

    def _write_experiment_info(self, experiment_info_path):
        # git, time, script
        git_hash = self.git.get_commit_hash()
        git_branch = self.git.get_branch_name()
        now = datetime.now()

        with open(experiment_info_path, 'w') as f:
            f.write('experiment time: {}/{}/{}/{}/{}\n'.format(now.year, now.month, now.day, now.hour, now.minute))
            f.write('git hash: ' + git_hash + '\n')
            f.write('git branch: ' + git_branch + '\n')
            f.write('command: ' + ' '.join(sys.argv) + '\n')

    def _write_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config.config, f)

    def _create_new_experiment_dir(self):
        """ create new experiment folder under self.experiment_dir
        :return: None
        """
        ls = os.listdir(self.experiment_dir)
        existing_exp_dirs = [d for d in ls if d.startswith('experiment')]
        if len(existing_exp_dirs) == 0:
            out = 'experiment1'
        else:
            inds = [int(d.lstrip('experiment')) for d in existing_exp_dirs]
            out = 'experiment'+str(max(inds)+1)

        self.new_experiment_dir = os.path.join(self.experiment_dir, out)
        os.mkdir(self.new_experiment_dir)
        return None


class Git(object):
    """ This object serves to get all git related info for you
    """
    def __init__(self):
        # check .git
        try:
            subprocess.call(['git', 'status'])
        except subprocess.CalledProcessError as e:
            raise ValueError("Probably not a git repository! ", e)

    def get_branch_name(self):
        branches = subprocess.check_output(['git', 'branch']).decode('utf-8').split('\n')
        branches = [b.strip() for b in branches if b]  # e.g. ['* classify-errors', 'fzq', 'master']
        branch = [b for b in branches if b.startswith('*')]
        return branch[0].lstrip('* ')

    def get_commit_hash(self):
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return git_hash


class BaseConfig(object):
    def __init__(self):
        self.config = dict()

    def update(self, config):
        """
        update self.config by config
        :param config: dict()
        :return:
        """
        # find keys are in config but not in self.config
        extra_keys = set(config.keys()) - set(self.config.keys())
        if len(extra_keys) > 0:
            raise ValueError("keys {} in config are not in Config.config".format(extra_keys))
        # update self.config by config
        else:
            self.config.update(config)


def parse_json(json_path):
    """parse object into a python dict"""
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out