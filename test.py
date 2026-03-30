"""
测试入口。

配置来源统一放在 experiment_config.py。
这个文件只负责：
1. 解析当前测试实验
2. 加载模型权重
3. 执行完整测试流程
"""

import os

import torch

from experiment_runtime import (
    build_active_test_bundle,
    create_test_setup,
    get_checkpoint_path,
    prepare_model_for_inference,
)
from testing_utils import ModelTester
from utils.config import Config


def main():
    """主测试函数。"""
    print(f"\n{'='*60}")
    print("小波包神经网络测试系统")
    print(f"{'='*60}")

    bundle = build_active_test_bundle()
    if bundle.device.type == "cpu":
        torch.backends.mkldnn.enabled = True
        print("⚙️ CPU benchmark mode: MKLDNN enabled")

    print(f"🎯 当前测试实验: {bundle.selection.mode} @ {bundle.selection.dataset}")
    Config.print_config_summary(
        bundle.dataset_config,
        bundle.model_config,
        test_config=bundle.test_config,
    )

    print("\n📊 Loading test dataset...")
    model, test_loader, class_names = create_test_setup(bundle)
    print(f"✅ Test dataset loaded: {len(test_loader.dataset)} samples")

    print("\n🏗️ Creating and loading model...")
    print(f"🎯 Using model type: {bundle.model_config.mode}")
    print(f"📊 架构描述: {Config.get_architecture_description(bundle.model_config.mode)}")
    print(f"📊 特征提取: {Config.get_feature_extraction_description(bundle.model_config.mode)}")

    checkpoint_path = get_checkpoint_path(bundle)
    if os.path.exists(checkpoint_path):
        loaded = False
        try:
            from utils.model_factory import ModelFactory

            loaded = ModelFactory.load_model_weights(model, checkpoint_path, bundle.device)
        except Exception as exc:
            print(f"❌ Failed to load model weights: {exc}")

        if loaded:
            print(f"✅ Model weights loaded: {checkpoint_path}")
        else:
            print("⚠️ Weight loading skipped, will use randomly initialized model for testing")
    else:
        print(f"⚠️ Model weight file does not exist: {checkpoint_path}")
        print("Will use randomly initialized model for testing")

    print("\n🔄 Preparing model for inference...")
    try:
        prepare_model_for_inference(model, bundle)
        print("✅ Inference preparation completed")
    except Exception as exc:
        print(f"⚠️ Inference preparation failed: {exc}")
        print("Will continue with the regular inference path")

    print("\n🧪 Starting model testing...")
    tester = ModelTester(
        model=model,
        test_loader=test_loader,
        test_config=bundle.test_config,
        dataset_config=bundle.dataset_config,
        model_config=bundle.model_config,
        device=bundle.device,
    )

    try:
        input_shape = (bundle.dataset_config.in_channels, bundle.dataset_config.input_length)
        test_results = tester.run_complete_test(input_shape, class_names)
        print("\n🎉 测试完成!")
        print(f"📊 测试精度: {test_results['accuracy'] * 100:.2f}%")
        print(f"📊 混淆矩阵精度: {test_results['confusion_matrix_accuracy']:.3f}")
        print(f"📊 模型参数量: {test_results['complexity_info']['params_str']}")
        print(f"📊 计算复杂度: {test_results['complexity_info']['flops_str']}")
        print(f"📊 模型大小: {test_results['complexity_info']['model_size_mb']:.2f} MB")
        print(f"📊 平均推理时间: {test_results['detailed_inference_times']['mean']:.4f} ms")
        print(f"💾 详细结果已保存至: {tester.results_dir}")
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        return
    except Exception as exc:
        print(f"\n❌ 测试过程中发生错误: {exc}")
        raise

    print("\n✅ 测试流程完成!")


if __name__ == "__main__":
    main()
