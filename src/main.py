import argparse
import platform
import sys

import psutil

from export_and_cpp_interface import *
from predict import *

sys.path.append(".")  # 把当前目录加入Python路径
import cv2
from generateImage import *
from definitions import *
from data_pipeline import *
from cv_baseline import *
from train import *
from universalFun import *


# ══════════════════════════════════════════════════════════════
#  生成样本
# ══════════════════════════════════════════════════════════════
def generate():
    for index in range(1, 12):
        base_dir = PROJECT_ROOT / "data" / "crop"
        files = list(base_dir.glob(f"{index}.*"))
        if not files:
            raise FileNotFoundError("没有找到 1.* 文件")
        input_path = str(files[0])
        output_path = PROJECT_ROOT / "data/generateImage"
        augment_image(input_path, output_path, 200, 200 * (index - 1))
    augment_image(PROJECT_ROOT / "data/other/0.png", output_path, 200, 200)


# ══════════════════════════════════════════════════════════════
#  cv_baseline
# ══════════════════════════════════════════════════════════════
def cv_baseline():
    # 配置字典
    cv_config = {
        "nominal_ar": 3.0,
        "min_tip_area": 200,
        "dark_thresh": 80,
        "px_per_um": 1.5,
        "min_calib_area": 500,
        "edge_thresh": 50,
        "min_mark_area": 100,
        "max_mark_area": 1000,
        "contrast_thresh": 30,
    }
    # 加载数据
    cvBase = CVBaseline(cv_config)
    image = cv2.imread(str(PROJECT_ROOT / "data/raw/auto_annotate" / "aug_0.*"))
    image = cv2.imread(PROJECT_ROOT / "data/template/tipTemplate.png")
    cvBase.run(image)
    folded_path = PROJECT_ROOT / "data/generateImage"
    save_path = PROJECT_ROOT / "data/annotated/auto_labeling.json"
    cvBase.generate_labels(folded_path, save_path)


# ══════════════════════════════════════════════════════════════
#  coco数据集可视化
# ══════════════════════════════════════════════════════════════
def COCO_vis():
    import fiftyone as fo
    import fiftyone.types as fot
    # 数据路径
    dataset_dir = PROJECT_ROOT/"data/generateImage"
    labels_path = PROJECT_ROOT / "data/annotated/auto_labeling.json"
    # 创建数据集
    dataset = fo.Dataset.from_dir(
        dataset_type=fot.COCODetectionDataset,
        data_path=dataset_dir,
        labels_path=labels_path,
    )
    print("dataset loaded")
    # 启动可视化界面
    '''   先关闭浏览器界面，否则进程报错  '''
    session = fo.launch_app(dataset)
    print("app launched")
    session.wait()
    '''
    cmd 关闭进程：
    tasklist | findstr mongod
    taskkill /F /PID 12345
    '''



# ══════════════════════════════════════════════════════════════
#  json分割
# ══════════════════════════════════════════════════════════════
def JSON_split():
    all_path = PROJECT_ROOT / "data/annotated/auto_labeling.json"
    split_path = PROJECT_ROOT / "data/raw/train/splits"
    split_dataset(all_path, split_path)


# ══════════════════════════════════════════════════════════════
#  模型训练
# ══════════════════════════════════════════════════════════════
def model_train():
    # 模型和训练参数配置
    config = DEFAULT_CONFIG.copy()
    config['data_root'] = PROJECT_ROOT / "data/raw/train"
    config['output_dir'] = PROJECT_ROOT / "experiments"
    # config['resume'] = PROJECT_ROOT/"experiments/exp_full_1783228941/last_phase2.pth"

    # 定义 命令行接口
    parser = argparse.ArgumentParser(description='GSG探针视觉系统训练')
    parser.add_argument('--mode', choices=['phase1', 'phase2', 'full', 'ablation'],
                        default='full', help='训练模式')
    parser.add_argument('--config', type=str, default=None, help='配置JSON路径')
    parser.add_argument('--resume', type=str, default=None, help='断点续训路径')
    parser.add_argument('--ablation', type=str, default=None,
                        help='单个消融实验名称（若指定则只跑一个）')
    args = parser.parse_args()
    # 如果指定了外部配置文件，则更新默认配置
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    # 如果指定了断点续训路径，则更新配置
    if args.resume:
        config['resume'] = args.resume
    # 训练模式指定
    if args.mode == 'ablation':
        assert ("开启消融实验")
        # if args.ablation:
        #     single_cfg = {**config, **ABLATION_CONFIGS[args.ablation],
        #                   'ablation_mode': args.ablation}
        #     trainer = Trainer(config)
        #     trainer = Trainer(single_cfg)
        #     trainer.phase1_pretrain()
        #     trainer.phase2_finetune()
        # else:
        #     run_ablation_study(config, config['output_dir'])
    else:
        # 创建模型
        trainer = Trainer(config)
        if args.mode in ('phase1', 'full'):
            trainer.phase1_pretrain()
        if args.mode in ('phase2', 'full'):
            trainer.phase2_finetune()


# ══════════════════════════════════════════════════════════════
#  模型预测
# ══════════════════════════════════════════════════════════════
def model_predict():
    config = DEFAULT_CONFIG.copy()
    config['pretrained'] = False
    config['pth_dir'] = str(PROJECT_ROOT / "experiments/exp_full_1783832251/best_phase2.pth")
    config['predict_data_dir'] = str(PROJECT_ROOT / "data/raw/predict_dataset")
    # config['predict_data_dir'] = str(PROJECT_ROOT / "data/raw/predict_image")
    predictor = Predictor(config)
    result_dataest = predictor.predict_dataset()
    # result_images = predictor.predict_images()


# ══════════════════════════════════════════════════════════════
#  模型导出
# ══════════════════════════════════════════════════════════════
def model_export():
    # 保存C++头文件
    out = Path('/home/claude/gsg_probe_system/cpp_interface')
    out.mkdir(exist_ok=True)
    ('gsg_probe_detector.h').write_text(CPP_HEADER)
    ('gsg_probe_detector.cpp').write_text(CPP_IMPL_SKETCH)
    ('CMakeLists.txt').write_text(CMAKE_CONTENT)
    print("C++接口文件已生成")


if __name__ == '__main__':

    ''' CV_baseline ----------------------------'''
    # generate()
    # cv_baseline()
    # COCO_vis()
    # JSON_split()

    ''' model ---------------------------------'''
    # model_train()
    model_predict()
    # model_export()
