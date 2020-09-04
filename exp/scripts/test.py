import yaml
import argparse
import numpy as np

from torchvision.utils import save_image
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

from template_lib.v2.config import update_parser_defaults_from_yaml, global_cfg
from template_lib.utils import update_config


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

update_parser_defaults_from_yaml(parser)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

config = update_config(config, global_cfg)

tt_logger = TestTubeLogger(
    save_dir=eval(config['logging_params']['save_dir']),
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
load_dict = torch.load(config.ckpt_path)
experiment.load_state_dict(load_dict['state_dict'])
experiment.cuda()
experiment.eval()
imgs = experiment.model.sample(num_samples=64, current_device=0)

save_image(imgs, filename=f'{args.tl_imgdir}/test.png', nrow=8, normalize=True, scale_each=True)

pass


