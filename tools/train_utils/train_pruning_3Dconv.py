import glob
import os


from torch.autograd import Variable
from torchvision import datasets, transforms

# from vgg import vgg
import numpy as np
import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from .optimization import build_optimizer, build_scheduler
from pcdet.models import load_data_to_gpu
import spconv.pytorch as spconv

def train_pruning_3Dconv1(model, optimizer, train_loader, train_set, MODEL, cfg, args, optim_cfg, start_epoch, total_epochs, ckpt_save_dir, PRUNING2D, Random):
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
    # from tools.sliming_pruning import sliming_pru
    # model = sliming_pru(model, batch)

    print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            total += m.weight.data.shape[0]

    pruned = 0
    cfg_layer = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm1d):
            weight_copy = m.weight.data.clone()
            # size = weight_copy.numel()
            size = m.weight.data.shape[0]
            thre_index = int(size * (1-args.percent))
            weights_sorted, _ = torch.sort(weight_copy.abs())  # 从小到大的顺序排列

            if Random:
                weight_copy = torch.rand_like(weight_copy)
                weights_sorted, _ = torch.sort(weight_copy.abs())
            thre = weights_sorted[thre_index]
            # mask = weight_copy.abs().gt(thre).float().cuda()  # 大于thre为true
            mask = weight_copy.abs().ge(thre).float().cuda()  # 大于等于thre为True
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_layer.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, torch.nn.MaxPool2d):
            cfg_layer.append('M')

    pruned_ratio = pruned / total

    print('Pre-processing Successful!')
    print(cfg_layer)

    from pcdet.models import build_network

    # MODEL['BACKBONE_3D']['NUM_FILTERS'] = [int(x * args.percent) for x in MODEL['BACKBONE_3D']['NUM_FILTERS']]
    MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] = int(MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] * args.percent)
    MODEL['BACKBONE_3D']['WIDTH'] = MODEL['BACKBONE_3D']['WIDTH'] * args.percent
    if PRUNING2D:
        MODEL['BACKBONE_2D']['WIDTH'] = MODEL['BACKBONE_2D']['WIDTH'] * args.percent
        MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] = int(MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] * args.percent)

    newmodel=build_network(model_cfg=MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    newmodel.cuda()
    # --------transfer weights-----------------------
    layer_id_in_cfg = 0
    start_mask = torch.ones(5)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        # import pdb;pdb.set_trace()
        if isinstance(m0, torch.nn.BatchNorm1d):  # 修改为BatchNorm1d
            # idx1 = torch.nonzero(end_mask).squeeze()  # 使用torch.nonzero替代np.argwhere
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, spconv.SparseConv3d) or isinstance(m0, spconv.SubMConv3d):  # 处理SparseConv3d和SubMConv3d
            # idx0 = torch.nonzero(start_mask).squeeze()
            # idx1 = torch.nonzero(end_mask).squeeze()
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # w = m0.weight.data[:, idx0, :, :, :].clone()
            w = m0.weight.data[:, :, :, :, idx0].clone()

            w = w[idx1, :, :, :, :].clone()
            m1.weight.data = w.clone()
        else:
            m1 = m0

    # ------------------- save checkpoint ------------------------------------------------------
    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % start_epoch)
    save_checkpoint(
        checkpoint_state(newmodel, optimizer, start_epoch), filename=ckpt_name,
    )
    print("Build new model successfully")
    print(newmodel)
    print("-----------------------Sparse3D Conv Prune End!")
    # import pdb;pdb.set_trace()
    return newmodel

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
