import open3d
import sys
import os
# Add the parent directory of the current file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from data_handler import get_dataset
from torch.utils.data import DataLoader
from models import PointNet2Generator
from pathlib import Path


def train(args):
    exist_check_path = str(args.save_path) + '/result.xyz'
    # print(exist_check_path)
    # if os.path.exists(exist_check_path):
    #     print(exist_check_path)
    #     print('---------------EXIST FILE! go to the next pcl data!')
    #     return 0
    
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    # device = torch.device('cpu')
    print(f'device: {device}')

    target_pc: torch.Tensor = util.get_input(args, center=False).unsqueeze(0).permute(0, 2, 1).to(device)
    if 0 < args.max_points < target_pc.shape[2]:
        indx = torch.randperm(target_pc.shape[2])
        target_pc = target_pc[:, :, indx[:args.cut_points]]

    data_loader = get_dataset(args.sampling_mode)(target_pc[0].transpose(0, 1), device, args)
    util.export_pc(target_pc[0], args.save_path / 'target.xyz',
                   color=torch.tensor([255, 0, 0]).unsqueeze(-1).expand(3, target_pc.shape[-1]))

    model = PointNet2Generator(device, args)
    # model.load_state_dict(torch.load(str(args.save_path / f'generators/pretrain_model{args.target_pretrain_weight}.pt')))
    ################## pretrain weight path
    model.load_state_dict(torch.load('_result/ablation/300/generators/model99_280.pt'))

    # -- Discard FC weights
    for name, param in model.named_parameters():
        if name.find("fc.") != -1:
            torch.nn.init.uniform_(param.data, -args.init_var, args.init_var)

    print(f'number of parameters: {util.n_params(model)}')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    train_loader = DataLoader(data_loader, num_workers=0,
                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    for i, (d1, d2) in enumerate(train_loader):
        d1, d2 = d1.to(device), d2.to(device)
        model.train()
        optimizer.zero_grad()
        d_approx = model(d2)
        loss = chamfer_distance(d_approx, d1, mse=args.mse)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f'{args.save_path.name}; iter: {i} / {int(len(data_loader) / args.batch_size)}; Loss: {util.show_truncated(loss.item(), 6)};')

        if i % args.export_interval == 0:
            util.export_pc(d_approx[0], args.save_path / f'exports/export_iter:{i}.xyz')
            util.export_pc(d1[0], args.save_path / f'targets/export_iter:{i}.xyz')
            util.export_pc(d2[0], args.save_path / f'sources/export_iter:{i}.xyz')
            torch.save(model.state_dict(), args.save_path / f'generators/model{i}.pt')


if __name__ == "__main__":
    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)
    train(args)

    # -- Debugging
    # import os
    # from pathlib import Path
    #
    # data_dir="my_data/PU1K_raw_meshes/"
    # result_dir="result/paper_exp/pu1k"
    # pc_name="02747177.c50c72eefe225b51cb2a965e75be701c"
    # mode = "density"
    # pc_file=data_dir + "/test/input_2048/input_2048/" + pc_name + ".xyz"# 2048
    # inference = False
    #
    # args.lr = 0.0005
    # args.iterations = 10010
    # args.export_interval = 600
    # args.pc = pc_file
    # args.init_var = 0.15
    # args.D1 = 512
    # args.D2 = 512
    # args.save_path = Path(result_dir + "/finetune1/" + pc_name + "/" + mode)
    # args.sampling_mode = mode
    # args.batch_size = 8
    # args.k = 10
    # args.p1 = 0.85
    # args.p2 = 0.2
    # args.force_normal_estimation = True
    # args.mse = True
    #
    # if not os.path.exists(args.save_path):
    #     Path.mkdir(args.save_path, exist_ok=True, parents=True)
    # if not inference:
    #     Path.mkdir(args.save_path / 'exports', exist_ok=True, parents=True)
    #     Path.mkdir(args.save_path / 'targets', exist_ok=True, parents=True)
    #     Path.mkdir(args.save_path / 'sources', exist_ok=True, parents=True)
    #     Path.mkdir(args.save_path / 'generators', exist_ok=True, parents=True)
    #
    # with open(args.save_path / ('inference_args.txt' if inference else 'args.txt'), 'w+') as file:
    #     file.write(util.args_to_str(args))
    #
    # train(args)