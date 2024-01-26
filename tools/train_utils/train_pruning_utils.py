import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from .optimization import build_optimizer, build_scheduler
from pcdet.models import load_data_to_gpu

def train_pruning_model(model, optimizer, train_loader, optim_cfg, start_epoch, total_epochs, ckpt_save_dir):
    extra_optim = extra_lr_scheduler = None
    if optim_cfg.get('EXTRA_OPTIM', None) and optim_cfg.EXTRA_OPTIM.ENABLED:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            extra_optim = build_optimizer(model.module, optim_cfg.EXTRA_OPTIM)
        else:
            extra_optim = build_optimizer(model, optim_cfg.EXTRA_OPTIM)

        # last epoch is no matter for one cycle scheduler
        extra_lr_scheduler, _ = build_scheduler(
            extra_optim, total_iters_each_epoch=len(train_loader), total_epochs=total_epochs,
            last_epoch=-1, optim_cfg=optim_cfg.EXTRA_OPTIM
        )
    # ----------------------  pruning ----------------------------------------------------------------

    # teacher_model = model
    dataloader_iter = iter(train_loader)
    batch = next(dataloader_iter)
    load_data_to_gpu(batch)
    # import pdb;pdb.set_trace()
    from tools.sliming_pruning import sliming_pru
    model = sliming_pru(model, batch)


    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % start_epoch)
    save_checkpoint(
        checkpoint_state(model, optimizer, start_epoch), filename=ckpt_name,
    )
    #return model

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
