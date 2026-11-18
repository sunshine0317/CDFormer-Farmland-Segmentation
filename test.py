import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import dataloader
import numpy as np
from PIL import Image
from utils.metrics import Evaluator
from network import UNet
from network.Deeplabv3_plus import deeplabv3
from network import HRNet
from network.PSPNet import Pspnet
from network.ABCNet import ABCNet
from network.CMTFNet import CMTFNet
from network import MCCA
from network import CGNet
from network import ENet
from network import DenseASPP
from network import SegNet
from network import BuildFormer
from network.UANet.UANet_final import UANet_final_Res50
from network import DSNet
from network.UNetFormer import UNetFormer
from network.DBBANet.bestmodel import DBBANet_Enhanced
from tqdm import tqdm

class hyper_parameters:
    def __init__(self):
        self.blocks = ['BOTTLENECK','BASIC','BASIC','BASIC']#[Bottleneck,BasicBlock, BasicBlock, BasicBlock]
        self.num_modules = [1, 1, 4, 3]#modules重复数 [重复1次的4个Bottleneck,重复1次的1个BasicBlock,重复4次的1个BasicBlock，重复3次的1个BasicBlock]
        self.num_branches = [1, 2, 3, 4]#分支数
        self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        self.num_channels = [[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]]#各条分支的通道数
        self.fuse_method = ['sum', 'sum', 'sum', 'sum']


def onehot_to_mask(semantic_map, palette):
    """
    将分割结果转换为彩色图像
    semantic_map: (H, W) 包含类别索引的数组
    palette: 调色板，每个类别对应一个RGB颜色
    """
    colour_codes = np.array(palette)
    # 确保索引在有效范围内
    semantic_map = np.clip(semantic_map.astype(np.uint8), 0, len(palette)-1)
    # 将类别索引映射到RGB颜色
    colored_mask = colour_codes[semantic_map]
    return colored_mask.astype(np.uint8)


        
def test(test_load, net, Eva, opt, device):
    print(f"🔥 测试函数使用设备: {device}")

    # 分割结果调色板：黑色(背景)，绿色(农田)
    palette2 = [[0,0,0],[0,128,0]]  # 类别0: 黑色, 类别1: 绿色
    net.train(False)
    net.eval()

    print(f"📊 开始处理 {len(test_load)} 个batch...")
    for batch_idx, (sample,filename) in enumerate(tqdm(test_load, desc="Testing")):
        inputs, mask = sample['image'], sample['label']
        X = inputs.to(device)
        Y = mask.to(device)

        if opt.model_name == 'ABCNet':
            output = net(X)[0]
        elif opt.model_name == 'UANet':
            output = net(X)[4]
        elif opt.model_name == 'DBBANet':
            output = net(X, training=False)
        elif opt.model_name in ['DBBANet_Light', 'DBBANet_MobileNet_PVT', 'DBBANet_ResNet_MiT']:
            output = net(X, training=False)
        else:
            output = net(X)
        pred = output.data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva.add_batch(target, pred)
        output = torch.argmax(output,dim = 1)

        # 保存所有分割结果为彩色图像
        for i in range(output.shape[0]):
            # 获取分割结果 (H, W)，包含类别索引
            pred_mask = (torch.squeeze(output[i])).data.cpu().numpy()

            # 转换为彩色图像 (H, W, 3)
            colored_mask = onehot_to_mask(pred_mask, palette2)

            # 保存为PNG图像
            final_savepath = opt.save_path + '/' + filename[i] + '.png'
            im = Image.fromarray(colored_mask, mode='RGB')  # 明确指定RGB模式
            im.save(final_savepath)

            # 可选：也保存一个索引图像用于后续分析
            # index_savepath = opt.save_path + '/' + filename[i] + '_index.png'
            # Image.fromarray(pred_mask.astype(np.uint8), mode='L').save(index_savepath)

        # 显示进度信息
        if (batch_idx + 1) % 20 == 0:
            print(f"📊 已处理: {batch_idx + 1} batches")

    IoU = Eva.Intersection_over_Union()
    F1Score = Eva.F1Score()
    Precision = Eva.Precision()
    Recall = Eva.Recall()
    OA = Eva.OA()

    print('****    [Test]IoU: %.4f, F1score: %.4f, Precision: %.4f, Recall: %.4f, OA:%.4f ****' \
          % ( IoU[1],F1Score,Precision,Recall,OA))
    string_print = 'IOU: %.4f, F1score: %.4f, Precision: %.4f, Recall: %.4f, OA:%.4f'\
                   % (IoU[1],F1Score,Precision,Recall,OA)
    with open(opt.txt_name, 'w') as f:
        f.write(string_print)
        f.write('\n')

def main():
    import argparse
    # using argparse ,we can execute python file using command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')  # GPU优化
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size') # GPU优化尺寸
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--model_name', type=str, default='DBBANet_Enhanced',
                        help='the model')
    parser.add_argument('--data_name', type=str, default='FarmSeg',
                        help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=2, 
                        help='')# we have 2 class here
    parser.add_argument('--save_path', type=str,
                        default=r'/root/lanyun-fs/Code2/FarmSeg2/test-output')
    parser.add_argument('--model_path', type=str,
                        default=r'/root/lanyun-fs/Code2/FarmSeg2/output-FarmSeg-VL-PSP/DBBANet_Enhanced_FarmSeg-VL_best.pth',
                        help='path to the trained model weights')
    parser.add_argument('--txt_name',type=str,default= '')

    opt = parser.parse_args()
    # 将分割图像和测试结果都保存到服务器目录
    opt.save_path = r'/root/lanyun-fs/Code2/FarmSeg2/test-output/' + opt.model_name + '/' + opt.data_name + '/'
    opt.txt_path ='./'+ opt.model_name + opt.data_name + '_test_result.txt'
    # opt.model_path 现在通过命令行参数设置，有默认值
    opt.txt_name = opt.save_path + opt.model_name + opt.data_name + '_test_result.txt'
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # 确保测试结果文本文件的目录也存在
    txt_dir = os.path.dirname(opt.txt_name)
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    print(f"📁 分割结果保存路径: {opt.save_path}")
    print(f"📄 测试结果文本路径: {opt.txt_name}")

    if opt.data_name == 'FarmSeg-VL':
        # FarmSeg-VL数据集专用调色板：黑色背景 + 红色农田
        palette = [[0,0,0],[255,0,0]]
        opt.test_img = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/test/img/'
        opt.test_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg-VL/test/label/'
    elif opt.data_name == 'FarmSeg':
        # 原始FarmSeg数据集调色板：黑色背景 + 红色 + 绿色农田
        palette = [[0,0,0],[128,0,0],[0,128,0]]
        opt.test_img = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/test/img/'
        opt.test_gt = r'/root/lanyun-fs/Code2/FarmSeg2/FarmSeg/test/label/'
    else:
        # 默认调色板
        palette = [[0,0,0],[128,0,0],[0,128,0]]

    # 强制使用GPU (因为调试确认GPU可用)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU不可用，使用CPU")
    print(f"设备: {device}")

    with torch.no_grad():
        Eva = Evaluator(opt.segclass)
        if opt.model_name =='UNet':
            net = UNet.UNetWithResnet50Encoder(n_classes=opt.segclass).to(device)
        elif opt.model_name =='DeeplabV3+':
            net = deeplabv3.DeepLab(num_classes=opt.segclass).to(device)
        elif opt.model_name == 'PSPNet':
            net = Pspnet(num_classes=opt.segclass).to(device)
        elif opt.model_name =='HRNet':
            hp = hyper_parameters()
            net = HRNet.HighResolutionNet(hp.blocks, hp.num_channels, hp.num_modules, hp.num_branches, hp.num_blocks,
                                        hp.fuse_method,opt.segclass).to(device)
        elif opt.model_name == 'ABCNet':
            net = ABCNet.ABCNet(3, opt.segclass).to(device)
        elif opt.model_name == 'CMTFNet':
            net = CMTFNet.CMTFNet(num_classes= opt.segclass).to(device)
        elif opt.model_name == 'MCCANet':
            net = MCCA.MCCA(num_class = opt.segclass).to(device)
        elif opt.model_name =='CGNet':
            net = CGNet.Context_Guided_Network(classes = opt.segclass).to(device)
        elif opt.model_name == 'DenseASPP':
            net = DenseASPP.DenseASPP(n_class = opt.segclass).to(device)
        elif opt.model_name =='ENet':
            net =ENet.ENet(n_classes = opt.segclass).to(device)
        elif opt.model_name =='SegNet':
            net = SegNet.SegNet(classes=opt.segclass).to(device)
        elif opt.model_name == 'BuildFormer':
            net = BuildFormer.BuildFormerSegDP(num_classes=opt.segclass).to(device)
        elif opt.model_name =='UANet':
            net = UANet_final_Res50(channel=32, num_classes=opt.segclass).to(device)
        elif opt.model_name =='DSNet':
            net = DSNet.DSNetLocalGuide(classes = opt.segclass).to(device)
        elif opt.model_name =='UNetFormer':
            net = UNetFormer(num_classes = opt.segclass, pretrained=False).to(device)
        elif opt.model_name =='DBBANet':
            net = DBBANet(num_class=opt.segclass).to(device)
            print('🚀 DBBANet initialized with ResNet50 backbone + GBC + Frequency Enhancement (默认启用)')
        elif opt.model_name == 'DBBANet_CTRCL':
            net = DBBANet_CTRCL(num_class=opt.segclass).to(device)
            print('🚀 DBBANet_CTRCL initialized with CNN and Transformer backbones')
        elif opt.model_name == 'DBBANet_Enhanced':
            net = DBBANet_Enhanced(num_class=opt.segclass).to(device)
            print('🚀 DBBANet_Enhanced initialized with Enhanced Edge Decoder + Combined Loss')
        elif opt.model_name == 'DBBANet_Light':
            # 轻量化版本: MobileNetV2 + MiT-B0
            net = create_lightweight_dbbanet(num_classes=opt.segclass).to(device)
        elif opt.model_name == 'DBBANet_MobileNet_PVT':
            # MobileNetV2 + PVT组合
            net = create_flexible_dbbanet(opt.segclass, 'mobilenetv2', 'pvt').to(device)
        elif opt.model_name == 'DBBANet_ResNet_MiT':
            # ResNet50 + MiT-B0组合
            net = create_flexible_dbbanet(opt.segclass, 'resnet50', 'mit_b0').to(device)
        elif opt.model_name == 'DCCLNet_ConDSeg_GBC':
            net = DCCLNet_ConDSeg_GBC(num_class=opt.segclass, use_gbc=True, gbc_strategy='full').to(device)
            model_info = net.get_model_info()
            print('🚀 DCCLNet_ConDSeg_GBC initialized with GBC strategy: full')
            if model_info['gbc_benefits']:
                print(f"   - GBC优势: {', '.join(model_info['gbc_benefits'])}")
        # 设置为评估模式
        net.eval()

        # 加载模型权重时也要考虑设备
        if device.type == 'cuda':
            net.load_state_dict(torch.load(opt.model_path), strict=False)
        else:
            net.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=False)
    
        test_load = dataloader.get_loader(opt.test_img, opt.test_gt, palette, opt.batchsize, opt.trainsize,  mode ='test',num_workers=8, shuffle=False, pin_memory=True)
        print("****    Start Testing!    ****")
        print("****    Test ",opt.model_name," on ",opt.data_name," with class ",opt.segclass ," ****")
        test(test_load, net, Eva, opt, device)

if __name__ == '__main__':
    main()