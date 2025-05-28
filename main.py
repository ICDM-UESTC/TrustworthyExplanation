import torch
import hydra
import warnings
from omegaconf import OmegaConf
from dataset import get_dataset
from dataloader import get_dataloader
from model import get_model
from explainer import get_explainer
from trainer import get_trainer
from collections import defaultdict
import numpy as np
import random
import pickle
from datetime import datetime
from torch_geometric.data import Batch
import shutil

warnings.filterwarnings('ignore', category=Warning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(cfg, cur_round=0, total_round=1, dataset=None):
    device = torch.device('cuda', index=cfg.device_id) if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(4)
    '''load dataloader'''
    dataloader = get_dataloader(dataset=dataset,
                                batch_size=getattr(cfg.method, cfg.dataset.dataset_name).batch_size)

    model = get_model(getattr(cfg.method, cfg.dataset.dataset_name)).to(device)
    explainer = get_explainer(cfg.method.method_name, getattr(cfg.method, cfg.dataset.dataset_name)).to(device)

    '''load trainer'''
    assert cfg.dataset.num_class == getattr(cfg.method, cfg.dataset.dataset_name).num_class
    assert cfg.dataset.multi_label == getattr(cfg.method, cfg.dataset.dataset_name).multi_label
    save_dir = cfg.save_dir
    trainer = get_trainer(method_name=cfg.method.method_name,
                          model=model,
                          explainer=explainer,
                          dataloader=dataloader,
                          cfg=getattr(cfg.method, cfg.dataset.dataset_name),
                          device=device,
                          save_dir=save_dir)
    print(trainer.method_name)

    if cfg.calculate_shd:
        trainer.calculate_shd(cfg.dataset.dataset_name, cfg.method.method_name,
                              ensemble_numbers=np.arange(total_round))
        exit()

    if cfg.test_by_sample_ensemble:
        trainer.test_by_sample_ensemble(cfg.dataset.dataset_name, cfg.method.method_name,
                                        ensemble_numbers=np.arange(total_round))
        exit()

    '''train'''
    trainer.train()
    '''test'''
    metrics = trainer.test()

    new_checkpoints_path = f'{trainer.checkpoints_path[:-4]}_{cur_round}.pth'
    shutil.copyfile(trainer.checkpoints_path, new_checkpoints_path)

    print(metrics)
    return metrics


@hydra.main(config_path='configs', config_name='global', version_base='1.3')
def main(cfg):
    return run(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_time', type=int, default=1, help='suggest 1 or 10')
    parser.add_argument('--dataset', type=str, default='ba_2motifs', help='{ba_2motifs, mr, benzene, mutag}')
    parser.add_argument('--method', type=str, default='gsat', help='{att, cal, size, gsat}')
    parser.add_argument('--test_by_sample_ensemble', action='store_true', help='')
    parser.add_argument('--calculate_shd', action='store_true', help='')
    parser.add_argument('--calculate_shd_multi', action='store_true', help='')

    args = parser.parse_args()

    accumulated_metrics = defaultdict(list)
    with hydra.initialize(config_path="configs", version_base='1.3'):
        cfg = hydra.compose(config_name="global", overrides=[f"dataset={args.dataset}", f"method={args.method}"])
        # print(OmegaConf.to_yaml(cfg))
        OmegaConf.set_struct(cfg, False)
        cfg.test_by_sample_ensemble = args.test_by_sample_ensemble if isinstance(
            args.test_by_sample_ensemble, bool) else False
        cfg.calculate_shd = args.calculate_shd if isinstance(args.calculate_shd, bool) else False
        '''load dataset'''
        dataset = get_dataset(dataset_dir=cfg.dataset.dataset_root,
                              dataset_name=cfg.dataset.dataset_name,
                              data_split_ratio=cfg.dataset.get('data_split_ratio', None))

        for i in range(args.run_time):
            set_seed(i)
            metrics = run(cfg, cur_round=i, total_round=args.run_time, dataset=dataset)
            for key, value in metrics.items():
                accumulated_metrics[key].append(value)
    average_metrics = {key: (np.mean(values), np.std(values)) for key, values in accumulated_metrics.items()}
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(average_metrics)
