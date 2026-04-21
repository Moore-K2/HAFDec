# 导入Python2/3兼容的核心语法（绝对导入、除法规则、print函数标准化）
from __future__ import absolute_import, division, print_function

# 导入系统操作库：用于文件/文件夹路径处理、系统环境交互
import os
# 导入Python解释器交互库：代码中未直接使用，为通用工程导入
import sys
# 导入文件检索库：用于批量查找指定格式的文件（如文件夹内所有.jpg）
import glob
# 导入命令行参数解析库：定义和读取运行脚本时传入的参数（如--image_path）
import argparse
# 导入数值计算核心库：用于数组操作、矩阵运算、保存npy格式文件
import numpy as np
# 导入PIL图像处理库：用于图像读取、缩放、格式转换（Python主流图像处理库）
import PIL.Image as pil
# 导入matplotlib基础库：用于颜色映射、数据可视化的基础配置
import matplotlib as mpl
# 导入matplotlib颜色映射库：提供预设色板（如magma，用于深度图伪彩色渲染）
import matplotlib.cm as cm

# 导入PyTorch核心库：用于张量操作、模型加载、设备管理（GPU/CPU）
import torch
# 导入TorchVision库：提供图像变换（ToTensor）和数据集工具（仅用transforms）
from torchvision import transforms, datasets

# 导入自定义网络模块：包含Lite-Mono编码器的定义
import networks
# 导入自定义层：包含视差图（disp）转深度图（depth）的核心函数
from layers import disp_to_depth
# 导入OpenCV库：Make3D数据集预处理需要用到
import cv2
# 导入堆排序算法库：代码中未使用，为冗余导入（保留原代码结构）
import heapq
# 导入PIL图像文件处理扩展模块：解决图像加载相关问题
from PIL import ImageFile

# 允许加载截断/损坏的图像文件：避免因图像文件不完整导致程序崩溃
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 导入Make3D适配的模型类
from networks import BiDepthDecoder, BiDepthDecoder2, LiteMono, ReMono, \
    mpvit_small, VitDepthDecoder, DepthDecoder, ReMonov2, Mono2ResnetEncoder, Mono2DepthDecoder, LiteMonov2,HAFDepthDecoder
# 导入scipy的mat文件读取模块（Make3D深度真值/文件名列表读取需要）
from scipy import io
# 导入进度条库（Make3D数据加载可视化）
from tqdm import tqdm


def load_make3d_data(main_path):
    """
    专门用于加载Make3D数据集并执行预处理的函数
    参数：
        main_path: Make3D数据集根路径（包含Test134、Gridlaserdata、make3d_test_files.txt）
    返回：
        images: 预处理后的图像列表（RGB格式，已裁剪）
        output_directory: 输出目录（Make3D根路径）
        filenames: 图像文件名列表（用于输出文件命名）
    """
    # 读取Make3D测试集文件列表
    test_file_path = os.path.join(main_path, "make3d_test_files.txt")
    assert os.path.exists(test_file_path), f"Make3D文件列表不存在：{test_file_path}"
    with open(test_file_path) as f:
        test_filenames = f.read().splitlines()
    # 处理文件名：移除前4个字符（适配Make3D格式）
    filenames = [x[4:] for x in test_filenames]
    output_directory = main_path  # Make3D输出目录设为数据集根路径

    # 初始化图像列表
    images = []
    # Make3D图像裁剪参数
    color_new_height = int(1704 / 2)  # 裁剪后图像高度

    # 加载并预处理每张图像
    print("正在加载并预处理Make3D数据集...")
    for filename in tqdm(filenames):
        # 拼接图像路径
        img_path = os.path.join(main_path, "Test134", f"img-{filename}.jpg")
        if not os.path.exists(img_path):
            print(f"警告：Make3D图像不存在，跳过 {img_path}")
            filenames.remove(filename)  # 同步移除无效文件名
            continue

        # 用PIL加载图像（兼容损坏的JPEG），转RGB
        try:
            image = pil.open(img_path).convert('RGB')
            image = np.array(image)  # 转为numpy数组方便裁剪
        except Exception as e:
            print(f"警告：图像加载失败，跳过 {img_path}，错误：{e}")
            filenames.remove(filename)
            continue

        # 图像裁剪：保留中间区域（去除上下无效边界）
        h, w, _ = image.shape
        start_h = int((h - color_new_height) / 2)
        end_h = start_h + color_new_height
        image = image[start_h:end_h, :, :]
        images.append(image)  # 添加到预处理后的图像列表

    assert len(images) > 0, "Make3D数据集加载失败：无有效图像"
    print(f"Make3D数据集加载完成，共{len(images)}张有效图像")
    return images, output_directory, filenames


# 定义命令行参数解析函数：解析运行脚本时传入的参数
def parse_args():
    # 创建参数解析器对象，设置脚本描述信息
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    # ========== 新增参数：数据集类型（kitti/make3d） ==========
    parser.add_argument('--dataset', type=str,
                        help='dataset type: kitti/make3d',
                        default="kitti",  # 默认KITTI数据集
                        choices=["kitti", "make3d"])

    # 添加必选参数：测试图像/图像文件夹的路径（Make3D时为数据集根路径）
    parser.add_argument('--image_path', type=str,
                        help='path to a test image/folder (kitti) or make3d root path (make3d)', required=False)

    # 添加必选参数：预训练模型权重文件夹路径（存放encoder.pth和depth.pth）
    parser.add_argument('--load_weights_folder', type=str,
                        help='path of a pretrained model to use',
                        )

    # 添加可选布尔参数：是否从.txt文件读取图像列表（适配KITTI数据集格式）
    parser.add_argument('--test',
                        action='store_true',  # 传入--test则该参数为True，否则False
                        help='if set, read images from a .txt file (for kitti)',
                        )

    # 添加可选参数：选择Lite-Mono模型规模（不同参数量，适配不同算力）
    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono",  # 默认使用标准版Lite-Mono
                        choices=[  # 限定模型类型，避免输入错误
                            "lite-mono",  # 标准版（参数量最大，精度最高）
                            "lite-mono-small",  # 小型版（参数量较小，速度更快）
                            "lite-mono-tiny",  # 微型版（参数量最小，适合边缘设备）
                            "lite-mono-8m"])  # 800万参数量版（平衡速度与精度）

    # 添加可选参数：指定图像扩展名（检索文件夹时过滤，默认jpg）
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder (for kitti)', default="jpg")

    # 添加可选布尔参数：是否禁用CUDA（强制使用CPU推理，默认启用GPU）
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    # ========== 新增/保留模型配置参数（原代码手动设置的参数） ==========
    parser.add_argument('--model_size', type=str,
                        help='model size: re-mono/re_mono_8m/re_mono_tiny/re_mono_small',
                        default="re_mono")
    parser.add_argument('--depth_encoder', type=str,
                        help='depth encoder class name: ReMono/LiteMono',
                        default="ReMono")
    parser.add_argument('--depth_decoder', type=str,
                        help='depth decoder class name: BiDepthDecoder/DepthDecoder',
                        default="BiDepthDecoder")

    # 解析命令行输入的参数并返回参数对象
    return parser.parse_args()


# 定义核心预测函数：实现单张/批量图像的深度估计推理
def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    # 断言校验：必须指定权重文件夹路径，否则抛出异常（必选参数校验）
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"
    # 断言校验：Make3D时必须指定image_path为数据集根路径
    if args.dataset == "make3d":
        assert args.image_path is not None and os.path.exists(args.image_path), \
            "Make3D数据集需要指定有效的根路径（--image_path）"

    # 自动选择推理设备：优先使用GPU（CUDA可用且未禁用），否则使用CPU
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")  # 设置为GPU设备
        print(f"使用GPU推理：{torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")  # 设置为CPU设备
        print("使用CPU推理")

    # 打印模型权重加载路径（日志提示，方便排查路径错误）
    print("-> Loading model from ", args.load_weights_folder)
    # 拼接编码器权重文件路径（encoder.pth：特征提取网络权重）
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    # 拼接深度解码器权重文件路径（depth.pth：视差图生成网络权重）
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    # 加载编码器权重文件（指定map_location避免GPU/CPU不兼容问题）
    encoder_dict = torch.load(encoder_path, map_location=device)
    # 加载深度解码器权重文件（同上）
    decoder_dict = torch.load(decoder_path, map_location=device)

    # 从编码器权重中提取模型训练时的图像尺寸（关键！推理时需对齐训练尺寸）
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']
    print(f"模型训练尺寸：{feed_width}x{feed_height}")

    # LOADING PRETRAINED MODEL
    # 打印日志：开始加载预训练编码器
    print("   Loading pretrained encoder")
    # 动态加载编码器类（适配ReMono/LiteMono等）
    encoder_d = eval(args.depth_encoder)

    if args.model_size in ["lite-mono", "re_mono", "lite-mono-small"]:
        encoder = encoder_d(model=args.model_size, height=feed_height, width=feed_width)
    elif args.model_size in ["monodepth2", "monodepth2_wp"]:
        encoder = Mono2ResnetEncoder()
    # 过滤并加载编码器权重（避免KeyError）
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder = encoder.to(device)
    encoder.eval()  # 切换到推理模式

    # 加载深度解码器
    print("   Loading pretrained decoder")
    depth_d = eval(args.depth_decoder)
    depth_decoder = depth_d(encoder.num_ch_enc, scales=range(3))
    # 过滤并加载解码器权重
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
    depth_decoder = depth_decoder.to(device)
    depth_decoder.eval()  # 切换到推理模式

    # ========== 核心判断：区分KITTI/Make3D数据集处理逻辑 ==========
    if args.dataset == "make3d":
        # Make3D数据集：调用专属加载/预处理函数
        images, output_directory, filenames = load_make3d_data(args.image_path)
        total_images = len(images)
    else:
        # KITTI数据集：保留原有的路径处理逻辑
        # 场景1：输入是单个图像文件，且未开启--test模式（单张图像推理）
        if os.path.isfile(args.image_path) and not args.test:
            paths = [args.image_path]
            output_directory = os.path.dirname(args.image_path)
        # 场景2：输入是.txt文件，且开启--test模式（读取KITTI格式的图像列表）
        elif os.path.isfile(args.image_path) and args.test:
            gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
            paths = []
            with open(args.image_path) as f:
                filenames = f.readlines()
                for i in range(len(filenames)):
                    filename = filenames[i]
                    line = filename.split()
                    folder = line[0]
                    if len(line) == 3:
                        frame_index = int(line[1])
                        side = line[2]
                    f_str = "{:010d}{}".format(frame_index, '.jpg')
                    image_path = os.path.join(
                        'kitti_data', folder, "image_0{}/data".format(side_map[side]), f_str)
                    paths.append(image_path)
            output_directory = os.path.dirname(args.image_path)
        # 场景3：输入是文件夹路径（批量推理文件夹内所有图像）
        elif os.path.isdir(args.image_path):
            paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
            output_directory = args.image_path
        # 场景4：路径无效（文件/文件夹不存在），抛出异常
        else:
            raise Exception("Can not find args.image_path: {}".format(args.image_path))
        total_images = len(paths)
        print("-> Predicting on {:d} test images (KITTI)".format(total_images))

    # PREDICTING ON EACH IMAGE IN TURN
    # 禁用梯度计算上下文管理器：节省内存、加速推理（推理阶段无需反向传播）
    # 保存至指定目录（Make3D专用命名）
    if args.output_pred_dir is not None:
        output_directory_pred = args.output_pred_dir
    else:
        dir_name = args.depth_encoder + '_' + args.depth_decoder + '_pred'
        output_directory_pred = os.path.join(output_directory, dir_name)

    if not os.path.exists(output_directory_pred):
        os.makedirs(output_directory_pred)

    with torch.no_grad():
        # ========== 区分数据集的推理循环 ==========
        if args.dataset == "make3d":
            # Make3D数据集推理循环
            for idx in tqdm(range(total_images), desc="Make3D推理进度"):
                image = images[idx]
                filename = filenames[idx]
                original_height, original_width = image.shape[:2]

                # Make3D图像预处理（适配模型输入）
                # 转换为PIL图像（方便缩放）→ 缩放到训练尺寸 → 转Tensor
                input_image = pil.fromarray(image)
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)  # 添加batch维度
                input_image = input_image.to(device)

                # 模型推理
                features = encoder(input_image)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]  # 提取0尺度视差图

                # 视差图恢复原始尺寸
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                # 保存结果（Make3D专用命名）
                output_name = f"img-{filename}"  # 与原始图像名一致
                # 视差图转深度图
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

                # 保存原图
                im = pil.fromarray(image)
                name_dest_im = os.path.join(output_directory_pred, "{}_img.jpeg".format(output_name))
                im.save(name_dest_im)

                # 保存npy格式视差数据
                name_dest_npy = os.path.join(output_directory_pred, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

                # 保存伪彩色深度图
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                name_dest_im = os.path.join(output_directory_pred, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                # # 打印进度（每10张打印一次，避免刷屏）
                # if (idx + 1) % 10 == 0:
                #     print(f"   Processed {idx + 1}/{total_images} images - saved to: {name_dest_im}")
        else:
            # KITTI数据集推理循环（保留原有逻辑）
            for idx, image_path in enumerate(tqdm(paths, desc="kitti推理进度")):
                # 跳过视差图文件：避免对已生成的_disp.jpg重复推理
                if image_path.endswith("_disp.jpg"):
                    continue

                # Load image and preprocess
                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # im = pil.fromarray(input_image)
                # name_dest_im = os.path.join(output_directory_pred, "{}_img.jpeg".format(output_name))
                # im.save(name_dest_im)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)
                disp = outputs[("disp", 0)]

                # 视差图恢复原始尺寸
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                output_name = os.path.splitext(os.path.basename(image_path))[0]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                name_dest_npy = os.path.join(output_directory_pred, "{}_disp.npy".format(output_name))
                # Saving numpy file
                if args.save_npy:
                    np.save(name_dest_npy, scaled_disp.cpu().numpy())

                # Saving colormapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                name_dest_im = os.path.join(output_directory_pred, "{}_disp.jpeg".format(output_name))
                im.save(name_dest_im)

                # 打印推理进度
                # print("   Processed {:d} of {:d} images - saved predictions to:".format(
                #     idx + 1, len(paths)))
                # print("   - {}".format(name_dest_im))
                # print("   - {}".format(name_dest_npy))

    # 打印推理完成提示
    print('-> Done!')


# 脚本入口：仅当直接运行该脚本时执行
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # ========== 配置示例1：测试Make3D数据集 ==========
    # args.dataset = "make3d"  # 指定为Make3D数据集
    # args.image_path = r'E:\datasets\make3d_test'  # Make3D数据集根路径
    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\re_mono_w14_wp'
    # args.model_size = "re_mono"
    # args.depth_encoder = 'ReMono'
    # args.depth_decoder = 'BiDepthDecoder'

    # ========== 配置示例2：测试KITTI数据集（原有配置） ==========
    # args.dataset = "make3d"  # 指定为KITTI数据集 /make3d
    # args.image_path = r'E:\datasets\make3d_test'
    args.save_npy = False
    args.dataset = "kitti"  # 指定为KITTI数据集 /make3d
    # args.dataset = "make3d"  # 指定为KITTI数据集 /make3d


    if args.dataset == "make3d":
        args.image_path = r'E:\datasets\make3d_test'
    elif args.dataset == "kitti":
        args.image_path = r'E:\datasets\kitti_data_test\test_img'
    else:
        args.image_path = r'E:\datasets\kitti_data_test\test_img'

    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\re_mono_w14_wp'
    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\re_mono_w29_p'
    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\hl_mono'
    args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\lite_mono_small'
    # args.model_size = "re_mono"  # model_size: re-mono # [re-mono, re_mono_8m,re_mono_tiny,re_mono_small]
    args.model_size = "lite-mono-small"  # model_size: re-mono # [lite_mono_small]
    # args.depth_encoder = 'ReMono'
    # args.depth_encoder = 'LiteMonov2'
    args.depth_encoder = 'LiteMono'
    # args.depth_decoder = 'BiDepthDecoder'
    # args.depth_decoder = 'HAFDepthDecoder'
    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\lite_mono'
    # args.model_size = "lite-mono"  # model_size: lite-mono
    # args.depth_encoder = 'LiteMono'
    args.depth_decoder = 'DepthDecoder'

    # args.load_weights_folder = r'E:\AI_proj\Lite-Mono\model\monodepth2'
    # args.model_size = "monodepth2"  # model_size: lite-mono
    # args.depth_encoder = 'Mono2ResnetEncoder'
    # args.depth_decoder = 'Mono2DepthDecoder'

    pretrain = True

    if pretrain:
        dir_name = args.depth_encoder + '_' + args.depth_decoder + '_pred'
    else:
        dir_name = args.depth_encoder + '_' + args.depth_decoder + '_wp_pred'
    if args.dataset == "make3d":
        args.output_pred_dir = rf'E:\datasets\make3d_test\{dir_name}' # make3d
    else:
        args.output_pred_dir = rf'E:\datasets\kitti_data_test\{dir_name}' # kitti


    # 执行核心预测函数
    test_simple(args)