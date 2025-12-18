"""
重构后的测试主函数
简洁清晰，只保留主要流程控制
"""

import torch
import os

# 导入重构后的模块
from utils.config import Config, TestConfig, DatasetConfig, ModelConfig
from utils.model_factory import ModelFactory
from utils.dataset_utils import DatasetLoader
from testing_utils import ModelTester


def main():
    """主测试函数"""
    
    # ==================== 配置参数 ====================
    # 测试模式配置
    MODE = 'wavelet_lite'  # Auto-modified by quick_test_all_models.py
                     # 'wavelet_traditional': 传统小波包变换 
                     # 'wavelet_lite': 轻量化小波包变换 
                     # 标准模型: 'lstm', 'gru', 'transformer', 'cnn', 'resnet'
                     # 轻量化模型: 'lstm_lite', 'gru_lite', 'transformer_lite', 'cnn_lite', 'resnet_lite'
                              
    # 小波参数（仅在小波模式下使用）
    WAVELET_TYPE = "db4"  # 小波基类型
    WAVELET_LEVELS = 3    # 传统小波分解层数
    DECOMPOSE_LEVELS = 3  # 可学习小波分解级数
    NUM_PARALLEL_GROUPS = 4  # 并行分解组数
    
    # 数据集配置
    DATASET_TYPE = 'UCIHAR'  # Auto-modified for quick testing

    # 数据集划分类型配置
    # "stratified": 分层分割（保持类别平衡）
    # "subject_dependent": 依赖于受试者（按用户分割）
    # "subject_independent": 独立于受试者（用户间混合分割）
    SPLIT_TYPE = 'stratified'  # 可以改为 'subject_dependent' 或 'subject_independent'

    # 测试参数
    BATCH_SIZE = 50  # 测试批次大小
    DEVICE = "cuda"  # "auto", "cuda", "cpu"
    NUM_INFERENCE_TESTS = 100  # 单样本推理测试次数
    TSNE_PERPLEXITY = 30  # t-SNE可视化的困惑度参数
    
    # MHEALTH数据集特定参数
    MHEALTH_STEP_SIZE = 64
    MHEALTH_EXCLUDE_NULL = True
    
    # ==================== 初始化配置 ====================
    print(f"\n{'='*60}")
    print(f"小波包神经网络测试系统")
    print(f"{'='*60}")
    
    # 设备配置
    device = Config.setup_device(DEVICE)
    
    # 数据集配置
    dataset_config = Config.get_dataset_config(DATASET_TYPE)
    if DATASET_TYPE == "MHEALTH":
        dataset_config.step_size = MHEALTH_STEP_SIZE
        dataset_config.exclude_null = MHEALTH_EXCLUDE_NULL

    # 设置数据集划分类型
    dataset_config.split_type = SPLIT_TYPE
    
    # 模型配置
    model_config = ModelConfig(
        mode=MODE,
        wavelet_type=WAVELET_TYPE,
        wavelet_levels=WAVELET_LEVELS,
        decompose_levels=DECOMPOSE_LEVELS,
        num_parallel_groups=NUM_PARALLEL_GROUPS
    )
    
    # 测试配置
    test_config = TestConfig(
        batch_size=BATCH_SIZE,
        num_inference_tests=NUM_INFERENCE_TESTS,
        tsne_perplexity=TSNE_PERPLEXITY
    )
    
    # 打印配置信息
    print(f"测试模式: {MODE}")
    print(f"数据集: {DATASET_TYPE}")
    print(f"设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"推理测试次数: {NUM_INFERENCE_TESTS}")
    if MODE == "wavelet_traditional":
        print(f"小波基类型: {WAVELET_TYPE}")
        print(f"分解层数: {WAVELET_LEVELS}")
    elif MODE in ["wavelet_learnable", "wavelet_lite"]:
        print(f"可学习小波包分解级数: {DECOMPOSE_LEVELS}")
        print(f"并行分解组数: {NUM_PARALLEL_GROUPS}")
    print(f"{'='*60}\n")
    
    # 打印数据集信息
    print(f"🎯 Using dataset: {DATASET_TYPE}")
    print(f"   - Input channels: {dataset_config.in_channels}")
    print(f"   - Number of classes: {dataset_config.num_classes}")
    print(f"   - Time window: {dataset_config.input_length} time steps")
    if DATASET_TYPE == "MHEALTH":
        print(f"   - 传感器: 胸部+右手腕 (加速度计+陀螺仪)")
        print(f"   - 过滤: 排除空值/过渡状态 (类别0)")
    
    # ==================== 数据加载 ====================
    print(f"\n📊 Loading test dataset...")
    
    # 创建测试数据加载器
    test_loader = DatasetLoader.create_test_loader(dataset_config)
    
    # 获取类别名称
    class_names = DatasetLoader.get_class_names(dataset_config)
    
    print(f"✅ Test dataset loaded: {len(test_loader.dataset)} samples")
    
    # ==================== 模型创建和加载 ====================
    print(f"\n🏗️ Creating and loading model...")
    
    # 创建模型
    model = ModelFactory.create_model(model_config.mode, dataset_config, model_config, device)
    
    # 打印模型信息
    print(f"🎯 Using model type: {MODE}")
    print(f"📊 架构描述: {Config.get_architecture_description(model_config.mode)}")
    print(f"📊 特征提取: {Config.get_feature_extraction_description(model_config.mode)}")
    
    # 加载模型权重
    model_checkpoint_path = Config.get_model_checkpoint_path(model_config.mode, dataset_config.name)
    
    if os.path.exists(model_checkpoint_path):
        try:
            ModelFactory.load_model_weights(model, model_checkpoint_path, device)
            print(f"✅ Model weights loaded: {model_checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to load model weights: {str(e)}")
            print("Will use randomly initialized model for testing")
    else:
        print(f"⚠️ Model weight file does not exist: {model_checkpoint_path}")
        print("Will use randomly initialized model for testing")

    # ==================== 重参数化切换 ====================
    # 对于可学习小波包模型，支持切换到部署模式以提升推理性能
    if MODE in ["wavelet_lite", "wavelet_traditional"] and hasattr(model, 'decomposer') and hasattr(model.decomposer, 'switch_to_deploy'):
        print(f"\n🔄 Switching model to deployment mode for better inference performance...")
        try:
            model.decomposer.switch_to_deploy()
            print("✅ Successfully switched to deployment mode")
            print("🚀 Inference performance will be significantly improved")
        except Exception as e:
            print(f"⚠️ Failed to switch to deployment mode: {str(e)}")
            print("Will continue with training mode")

    # ==================== 模型测试 ====================
    print(f"\n🧪 Starting model testing...")
    
    # 创建测试器
    tester = ModelTester(
        model=model,
        test_loader=test_loader,
        test_config=test_config,
        dataset_config=dataset_config,
        model_config=model_config,
        device=device
    )
    
    # 执行完整测试
    try:
        input_shape = (dataset_config.in_channels, dataset_config.input_length)
        test_results = tester.run_complete_test(input_shape, class_names)
        
        # 打印测试结果摘要
        print(f"\n🎉 测试完成!")
        print(f"📊 测试精度: {test_results['accuracy'] * 100:.2f}%")
        print(f"📊 混淆矩阵精度: {test_results['confusion_matrix_accuracy']:.3f}")
        print(f"📊 模型参数量: {test_results['complexity_info']['params_str']}")
        print(f"📊 计算复杂度: {test_results['complexity_info']['flops_str']}")
        print(f"📊 模型大小: {test_results['complexity_info']['model_size_mb']:.2f} MB")
        print(f"📊 平均推理时间: {test_results['detailed_inference_times']['mean']:.4f} ms")
        
        results_dir = tester.results_dir
        print(f"💾 详细结果已保存至: {results_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
        return
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        raise e
    
    print(f"\n✅ 测试流程完成!")


if __name__ == "__main__":
    main()