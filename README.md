# FINDER：Python实现

这个仓库包含了在论文Fan, C., Zeng, L., Sun, Y and Liu Y-Y. [Finding key players in complex networks through deep reinforcement learning](https://www.nature.com/articles/s42256-020-0177-2.epdf?sharing_token=0CAxnrCP1THxBEtK2mS5c9RgN0jAjWel9jnR3ZoTv0O3ej6g4eVo3V4pnngJO-QMH375GbplyUstNSGUaq-zMyAnpSrZIOiiDvB0V_CqsCipIfCq-enY3sK3Uv_D_4b4aRn6lYXd8HEinWjLNM42tQZ0iVjeMBl6ZRA7D7WUBjM%3D). Nat Mach Intell (2020).中提出的FINDER模型的纯Python实现。原始实现可在[此链接](https://github.com/FFrankyy/FINDER)找到。

## 系统要求

### 软件依赖性

用户应首先安装以下软件包。在推荐的硬件配置下，这些软件包的安装时间约为5分钟。具体的软件版本如下：

- networkx==2.3
- numpy==1.17.3
- pandas==0.25.2
- scipy==1.3.1
- tensorflow-gpu==1.14.0
- tqdm==4.36.1

### 硬件要求

FINDER模型需要一台具有足够RAM和GPU支持的标准计算机来支持用户定义的操作。对于最低性能要求，建议配置为：

- RAM：至少4GB
- GPU：至少16GB

为了获得最佳性能，我们推荐以下配置的计算机：

- RAM：16GB以上
- CPU：4核以上，3.3GHz以上/核
- GPU：16GB以上

## 运行说明

### 训练模型

使用以下命令在指定GPU上训练模型：

```
CUDA_VISIBLE_DEVICES=gpu_id python FINDER.py
```

超参数存储在`configs/FINDER_ND.yaml`中，用户可以修改这些参数进行模型调优。

### 测试真实数据
使用以下命令在CPU上测试真实数据（不使用GPU）：

```
CUDA_VISIBLE_DEVICES=-1 python testReal.py
```

使用训练好的模型（存储在./models目录下），您可以复现论文中报告的结果。
