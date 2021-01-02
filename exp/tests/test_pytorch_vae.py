import os
import subprocess
import sys
import unittest
import argparse

from template_lib.utils.config import parse_args_and_setup_myargs, config2args
from template_lib.examples import test_bash
from template_lib import utils
from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
  start_cmd_run


class Testing_PyTorch_VAE(unittest.TestCase):

  def test_train_vae(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export TIME_STR=1
        export PYTHONPATH=./PyTorch_VAE_lib:./
        python -c "from exp.tests.test_pytorch_vae import Testing_PyTorch_VAE;\
          Testing_PyTorch_VAE().test_train_vae()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/configs/pytorch_vae.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python PyTorch_VAE_lib/run.py -c PyTorch_VAE_lib/configs/vae.yaml
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass

  def test_train_wae(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export TIME_STR=1
        export PYTHONPATH=./PyTorch_VAE_lib:./
        python -c "from exp.tests.test_pytorch_vae import Testing_PyTorch_VAE;\
          Testing_PyTorch_VAE().test_train_vae()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/configs/pytorch_wae.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python PyTorch_VAE_lib/run.py -c PyTorch_VAE_lib/configs/wae_mmd_imq.yaml
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass
  def test_test_vae(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export TIME_STR=1
        export PYTHONPATH=./PyTorch_VAE_lib:./
        python -c "from exp.tests.test_pytorch_vae import Testing_PyTorch_VAE;\
          Testing_PyTorch_VAE().test_test_vae()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/configs/pytorch_vae.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python exp/scripts/test.py -c PyTorch_VAE_lib/configs/vae.yaml
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass
  def test_test_wae(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export TIME_STR=1
        export PYTHONPATH=./PyTorch_VAE_lib:./
        python -c "from exp.tests.test_pytorch_vae import Testing_PyTorch_VAE;\
          Testing_PyTorch_VAE().test_test_vae()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/configs/pytorch_wae.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python exp/scripts/test.py -c PyTorch_VAE_lib/configs/wae_mmd_imq.yaml
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass

# from template_lib.v2.config import update_parser_defaults_from_yaml
# update_parser_defaults_from_yaml(parser)