"""
训练入口。

配置来源统一放在 experiment_config.py。
这个文件只负责：
1. 解析当前训练实验
2. 构造数据和模型
3. 调用 Trainer 执行训练
"""

import sys

from experiment_runtime import build_active_train_bundle, create_training_setup
from training_utils import Trainer
from utils.config import Config
from utils.dataset_utils import DatasetLoader


def main():
    """主训练函数。"""
    print(f"\n{'='*60}")
    print("小波包神经网络训练系统")
    print(f"{'='*60}")

    bundle = build_active_train_bundle()
    print(f"🎯 当前训练实验: {bundle.selection.mode} @ {bundle.selection.dataset}")
    Config.print_config_summary(
        bundle.dataset_config,
        bundle.model_config,
        training_config=bundle.training_config,
    )

    print("\n📊 Loading dataset...")
    model, train_loader, val_loader = create_training_setup(bundle)
    DatasetLoader.print_dataset_info(bundle.dataset_config)

    print("\n🏗️ Creating model...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数数量: {total_params:,}")
    print(f"📊 架构描述: {Config.get_architecture_description(bundle.model_config.mode)}")
    print(f"📊 特征提取: {Config.get_feature_extraction_description(bundle.model_config.mode)}")

    print("\n🚀 Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=bundle.training_config,
        dataset_config=bundle.dataset_config,
        model_config=bundle.model_config,
        device=bundle.device,
    )

    try:
        training_results = trainer.train()
        checkpoint_path = Config.get_model_checkpoint_path(
            bundle.model_config.mode,
            bundle.dataset_config.name,
        )
        print("\n🎉 训练完成!")
        print(f"📊 全局最佳验证精度: {training_results['best_acc']:.2f}%")
        print(f"📊 最后5轮最佳验证精度: {training_results['best_acc_last_epochs']:.2f}%")
        print(f"💾 最佳模型已保存至: {checkpoint_path}")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ 训练过程中发生错误: {exc}")
        raise

    print("\n✅ 训练流程完成!")


if __name__ == "__main__":
    main()
