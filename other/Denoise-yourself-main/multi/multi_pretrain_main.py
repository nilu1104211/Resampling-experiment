import open3d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from multi_data_handler import get_multi_dataset
from torch.utils.data import DataLoader
from models import PointNet2Generator

############ noise3 multi pretrain
def train(args):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    # device = torch.device('cpu')
    print(f'device: {device}')
    # multi data setting
    data_path = '../self_sample_data/my_data/PU1K_raw_meshes/train/train_gt_pc/300' # Use the entire dataset for both 2048 and 10k pretraining

    noise_path = data_path + '/0.002'
    noise_file = os.listdir(noise_path)
    data_file = os.listdir(data_path)
    data_file.remove('0.002')
   
   
    # model setting
    model = PointNet2Generator(device, args)
    # model = torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load("_result/ablation/noise_level_multi/0.006/pretrain/generators/model94_890.pt"))
    print(f'number of parameters: {util.n_params(model)}')
    model.initialize_params(args.init_var)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()
 
    noise_pc_lst = [] # Create a list in the format required by get_dataset
    clean_pc_lst = []

    for path in data_file:
        args.pc = data_path+'/'+path
        args.noise_pc = noise_path+'/'+path # args.sampling_mode = 'denoising'
        noise_pc, clean_pc = util.get_input_exp(args, center=False)
        noise_pc = noise_pc.unsqueeze(0).permute(0, 2, 1).to(device)
        clean_pc = clean_pc.unsqueeze(0).permute(0, 2, 1).to(device)
        if 0 < args.max_points < noise_pc.shape[2]:
            indx = torch.randperm(noise_pc.shape[2])
            noise_pc = noise_pc[:, :, indx[:args.cut_points]]
        if 0 < args.max_points < clean_pc.shape[2]:
            indx = torch.randperm(clean_pc.shape[2])
            clean_pc = clean_pc[:, :, indx[:args.cut_points]]
        # print(noise_pc)
        noise_pc_lst.append(noise_pc)
        clean_pc_lst.append(clean_pc)
        # print('noise')
        # print(noise_pc.shape)
        # print('clean')
        # print(clean_pc.shape)

    all_data = [get_dataset2("denoising")(noise_pc_lst[pc_idx][0].transpose(0, 1), clean_pc_lst[pc_idx][0].transpose(0, 1), device, args) for pc_idx in range(len(clean_pc_lst))] # 1 2 3 0
    data_len = len(all_data)
    # print('----')
    # print(data_len)
    epochs = 100
    # iterations = int(args.iterations/args.batch_size)
    # print(iterations)
    for epoch in range(epochs):
        for data_idx in range(data_len):
            # print(data_idx)
            current_data = all_data[data_idx]
            # print("current data")
            # print(current_data.shape)
            train_loader = DataLoader(current_data, num_workers=0,
                            batch_size=args.batch_size, shuffle=False, drop_last=False)
            # print('train loader')
            # print(train_loader)
            for i, (d1, d2) in enumerate(train_loader): # 1
                if i==1:
                    break
                # print('train_loader i')
                # print(i)
                # print(d2)
                # print(d2)
                d1, d2 = d1.to(device), d2.to(device)
                model.train()
                optimizer.zero_grad()
                # print(d2.shape)
                d_approx = model(d2)
                # print(d_approx.shape)
                loss = chamfer_distance(d_approx, d1, mse=args.mse)
                loss.backward()
                optimizer.step()

                if data_idx % 10 == 0:
                    print(f'{args.save_path.name}; epoch: {epoch}; iter: {data_idx}; Loss: {util.show_truncated(loss.item(), 6)};')

                if data_idx == 280:
                    # torch.save(model.state_dict(), args.save_path / f'generators/epoch/model{epoch}.pt')
                # if data_idx % args.export_interval == 0:
                    util.export_pc(d_approx[0], args.save_path / f'exports/export_iter:{epoch}_{data_idx}.xyz')
                    util.export_pc(d1[0], args.save_path / f'targets/export_iter:{epoch}_{data_idx}.xyz')
                    util.export_pc(d2[0], args.save_path / f'sources/export_iter:{epoch}_{data_idx}.xyz')
                    torch.save(model.state_dict(), args.save_path / f'generators/model{epoch}_{data_idx}.pt')


if __name__ == "__main__":
    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)
    train(args)
