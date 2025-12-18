import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """
    1D ResNet基础块，用于时间序列数据
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """
    1D ResNet瓶颈块，用于更深的网络
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        
        # 1x1卷积降维
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 3x3卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1卷积升维
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StandardResNet(nn.Module):
    """
    标准ResNet模型，用于时间序列分类
    具有更深的网络结构和更多的参数
    """
    def __init__(self, in_channels, num_classes, input_length, verbose=False):
        super(StandardResNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.verbose = verbose
        
        # 优化的基础通道数 (从128降到32，减少参数量)
        base_channels = 32
        self.inplanes = base_channels
        
        # 初始卷积层 - 优化的卷积核和通道数
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=15, 
                              stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 优化的ResNet层 (从[3,4,6,3]改为[2,2,2,2]，减少参数量)
        self.layer1 = self._make_layer(BasicBlock1D, base_channels, 2)
        self.layer2 = self._make_layer(BasicBlock1D, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, base_channels * 8, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 简化的分类器 (从3层改为2层，减少参数量)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        if verbose:
            self._print_model_info()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*50}")
        print(f"标准ResNet分类器模型架构")
        print(f"{'='*50}")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*50}")

    def forward(self, x):
        # x: [batch_size, in_channels, seq_len]
        
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class LightweightBasicBlock1D(nn.Module):
    """
    轻量化ResNet基本块的1D版本，减少参数量
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LightweightBasicBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 使用更小的卷积核和减少通道数
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LightweightResNet(nn.Module):
    """
    轻量化ResNet的1D版本，减少参数量和计算复杂度
    """
    def __init__(self, in_channels, num_classes, input_length, verbose=False):
        super(LightweightResNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_length = input_length
        self.verbose = verbose
        
        # 调整初始通道数到14，达到目标参数量
        base_channels = 14
        self.inplanes = base_channels
        
        # 初始卷积层 - 调整通道数
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层 - 优化层数配置 (调整为[2,1,1,1]，达到目标参数量)
        self.layer1 = self._make_layer(LightweightBasicBlock1D, base_channels, 2)
        self.layer2 = self._make_layer(LightweightBasicBlock1D, base_channels * 2, 1, stride=2)
        self.layer3 = self._make_layer(LightweightBasicBlock1D, base_channels * 4, 1, stride=2)
        self.layer4 = self._make_layer(LightweightBasicBlock1D, base_channels * 8, 1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if verbose:
            self._print_model_info()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n==================================================")
        print(f"轻量化ResNet分类器模型架构")
        print(f"==================================================")
        print(f"输入通道数: {self.in_channels}")
        print(f"输入长度: {self.input_length}")
        print(f"输出类别数: {self.num_classes}")
        print(f"总参数量: {total_params:,}")
        print(f"==================================================")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    print("=== ResNet模型测试 ===")
    
    # 测试参数
    batch_size, in_channels, seq_len = 4, 6, 128
    num_classes = 6
    x = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 测试标准ResNet
    print("\n1. 测试标准ResNet:")
    standard_resnet = StandardResNet(in_channels, num_classes, seq_len, verbose=True)
    out_standard = standard_resnet(x)
    print(f"标准ResNet输出形状: {out_standard.shape}")
    
    # 测试轻量级ResNet
    print("\n2. 测试轻量级ResNet:")
    lightweight_resnet = LightweightResNet(in_channels, num_classes, seq_len, verbose=True)
    out_light = lightweight_resnet(x)
    print(f"轻量级ResNet输出形状: {out_light.shape}")
    
    print("\n=== ResNet模型测试完成 ===")