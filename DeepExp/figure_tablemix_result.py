import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.onnx.symbolic_opset11 import unsqueeze

from utils.sweeper import Sweeper

#plt.style.use('seaborn-ticks')
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Set font family, bold, and font size
# font = {'family':'normal', 'weight':'normal', 'size': 12}
font = {'size': 15}
matplotlib.rc('font', **font)
# Avoid Type 3 fonts in matplotlib plots: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils.helper import make_dir
from utils.plotter import read_file, get_total_combination, symmetric_ema
import os

#table_latex_file_path = os.path.join(".\\figures", "latex_tab.txt")

class ExperimentData:
    """实验数据类"""

    def __init__(self, env_name, data_dict):
        self.env_name = env_name
        self.data_dict = data_dict  # {method_name: {'x_mean': [...], 'y_mean': [...]}}

class mycolorclass:
    def __init__(self):
        self.black = '#000000'
        self.blue = '#090EE7'
        self.red = '#ff0000'
        self.green = '#66ff66'
        self.yellow = '#ffcc00'
        self.pink = '#ff00ff'
        self.brown = '#990000'
        self.molvse = '#385723'
        self.qianlanse = '#39b6c7'
        self.purple = '#9900ff'
        self.lianghuangse = '#EEF30D'


def plot_baseline_figure5(data_list, baseline_methods, our_methods, figsize=(16, 10)):  # 稍微调高高度
    """
    绘制实验对比图 - 5图倒等腰梯形布局
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    fig = plt.figure(figsize=figsize)

    # 创建 2行 6列 的网格
    # 第一行 3个子图，每个占 2 列: (0,0-1), (0,2-3), (0,4-5)
    # 第二行 2个子图，每个占 2 列，居中则占用: (1,1-2), (1,3-4)
    gs = gridspec.GridSpec(2, 6, figure=fig)
    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.1,
        top=0.85,  # 留出更多空间给顶部的图例
        wspace=1.0,  # 增加间距
        hspace=0.4
    )

    # 定义 5 个子图的位置坐标
    # 前三个在第一行，后两个在第二行居中
    ax_positions = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],  # 第一行
        gs[1, 1:3], gs[1, 3:5]  # 第二行居中
    ]

    axes = []
    for pos in ax_positions:
        axes.append(fig.add_subplot(pos))

    # 设置颜色方案 (保持不变)
    mycolor = mycolorclass()
    color_dict = {
        "OrderDQN": mycolor.brown,
        "DQN": mycolor.blue,
        "DDQN": mycolor.green,
        "AveragedDQN": mycolor.pink,
        "MaxminDQN": mycolor.molvse,
        "WeightedDQN": mycolor.qianlanse,
        "EBDQN": mycolor.purple,
        "ACCDDQN": mycolor.lianghuangse,
    }

    method2lable = {
        "OrderDQN": 'Order DQN', "DQN": 'DQN', "DDQN": 'DDQN',
        "AveragedDQN": 'Averaged DQN', "MaxminDQN": 'Maxmin DQN',
        "WeightedDQN": 'Weighted DDQN', "EBDQN": 'EBDQN', "ACCDDQN": 'ACC DDQN',
    }

    # 遍历每个环境 (data_list 长度应为 5)
    for env_idx, data in enumerate(data_list):
        if env_idx >= 5: break  # 确保只画5个

        env_name = data.env_name
        ax = axes[env_idx]

        # --- 原始逻辑：坐标轴与范围设置 ---
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                      ['0', '2', '4', '6', '8', '10'])
        ax.grid()

        # --- 原始逻辑：计算 Best Method ---
        best_our_method = None
        best_our_value = -float('inf')
        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                start_index = int(len(y_mean) * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()
                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # --- 原始逻辑：绘制曲线 ---
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']
            ax.plot(x_mean, y_mean, color=color_dict[method], linewidth=1.5, label=method)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        if best_our_method:
            x_mean = data.data_dict[best_our_method]['x_mean']
            y_mean = data.data_dict[best_our_method]['y_mean']
            y_ci = data.data_dict[best_our_method]['y_ci']
            ax.plot(x_mean, y_mean, color='red', linewidth=1.5, label="QMDDQN(Our Best)")
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor='red', alpha=0.5)

        # y轴标签只加在每行最左侧
        if env_idx == 0 or env_idx == 3:
            ax.set_ylabel('Average Return', fontsize=20)

        figcharidx = chr(ord('a') + env_idx)
        ax.set_xlabel(f'Steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)

    # --- 统一图例绘制 ---
    legend_handles = []
    legend_labels = []
    for method in baseline_methods:
        label = method2lable[method]
        line, = plt.plot([], [], color=color_dict[method], linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)

    our_label = "QMDDQN(Our Best)"
    line, = plt.plot([], [], color='red', linewidth=3, label=our_label)
    legend_handles.append(line)
    legend_labels.append(our_label)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=1.8,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),  # 置于顶部
               columnspacing=3.4,
               ncol=min(len(legend_handles), 5),
               fontsize=18,
               frameon=True)

    fig.savefig(f'./figures/deepbaseline3_2.png', dpi=300)
    return fig, axes


def plot_baseline_figure(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，3行2列
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.07,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.1,  # 底边距 (默认0.11)
        top=0.89,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.35  # 垂直间距 (默认0.2)
    )
    axes[1, 2].set_visible(False)

    # 设置颜色方案
    mycolor = mycolorclass()
    color_dict = {
        "OrderDQN": mycolor.brown,
        "DQN": mycolor.blue,
        "DDQN": mycolor.green,
        "AveragedDQN": mycolor.pink,
        "MaxminDQN": mycolor.molvse,
        "WeightedDQN": mycolor.qianlanse,
        "EBDQN": mycolor.purple,
        "ACCDDQN": mycolor.lianghuangse,
    }

    method2lable = {
        "OrderDQN": 'Order DQN',
        "DQN": 'DQN',
        "DDQN": 'DDQN',
        "AveragedDQN": 'Averaged DQN',
        "MaxminDQN": 'Maxmin DQN',
        "WeightedDQN": 'Weighted DDQN',
        "EBDQN": 'EBDQN',
        "ACCDDQN": 'ACC DDQN',

    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 3
        col = env_idx % 3
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-300, -100)
        #Puckworld ylim
        if env_name == "PuckWorld":
            #ax.set_ylim(-2000, -1000)
            pass
        # if env_name == "SpaceInvaders":
        #     ax.set_ylim(20, 45)
        if env_name == "Pixelcopter":
            pass
            #ax.set_ylim(10, 30)
        if env_name == "Seaquest":
            #ax.set_ylim(-100, -100)
            pass

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(20)  # 调整指数字体大小
        #ax.yaxis.get_offset_text().set_weight('bold')  # 设置指数加粗
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method

            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"TMDDQN(Our Best)"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor='red', alpha=0.5)

        if env_idx in [0,3]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)
        #ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    legend_handles = []
    legend_labels = []
    for method in baseline_methods:
        label = method2lable[method]
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)
    label = f"TMDDQN(Our Best)"
    line, = plt.plot([], [], color=mycolor.red,
                     linewidth=3, label=label)
    legend_handles.append(line)
    legend_labels.append(label)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=1.8,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=3.4,
               ncol=5,
               fontsize=18,
               frameon=True)

    fig.savefig(f'./figures/deepbaseline3_2.png', dpi=300)



    return fig, axes

def plot_para_figure(data_list, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行N列
    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.07,  # 左边距 (默认0.125)
        right=0.98,  # 右边距
        bottom=0.1,  # 底边距 (默认0.11)
        top=0.93,  # 顶边距
        wspace=0.25,  # 水平间距 (默认0.2)
        hspace=0.35  # 垂直间距 (默认0.2)
    )
    axes[1, 2].set_visible(False)

    # 设置颜色方案
    mycolor = mycolorclass()
    color_list = [
         mycolor.red,
        mycolor.green,
        mycolor.brown,
        mycolor.blue,
        mycolor.black,
        mycolor.molvse,
        mycolor.qianlanse,
        mycolor.pink,
        mycolor.purple,
        mycolor.yellow,
    ]


    color_dict = {
        "MIX8":   mycolor.red ,
        "MIX9": mycolor.green ,
        "MIX20": mycolor.brown ,
        "MIX11": mycolor.blue ,
        "MIX12": mycolor.black ,
        "MIX13": mycolor.molvse ,
        "MIX14": mycolor.qianlanse ,
        'MIX15': mycolor.red,
        'MIX16': mycolor.green,
        'MIX17': mycolor.brown,

    }
    color_dict = {}
    for i, method in enumerate(our_methods):
        color_dict[method] = color_list[i]

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 3
        col = env_idx % 3
        ax = axes[row, col]

        #MC env ylim
        if env_name == "Asterix":
            ax.set_ylim(9, 14.5)
        if env_name == "Breakout":
            ax.set_ylim(10, 20.5)
        if env_name == "Seaquest":
            ax.set_ylim(5, 16)
        if env_name == "SpaceInvaders":
            ax.set_ylim(40, 60)
        if env_name == "Pong":
            ax.set_ylim(-5, 0)




        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 绘制我们的方法

        for method in our_methods:
            # if method[-1] not in ["2","3", "4", "5", "6","7", "8"]:
            #     continue
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        if env_idx in [0,3]:
            ax.set_ylabel('Average Return', fontsize=20)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)


    legend_handles = []
    legend_labels = []
    for method in our_methods:
        # if method == 'M-DDQN0' or method == 'M-DDQN1':
        #     rho = float(method[-1])
        # else:
        #     rho = float(method[-3:])
        # label = r"$\rho=$" + f'{rho:.1f}'
        label = r"$M$="+method[3:]

        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=2,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),  # 在图形正上方
               columnspacing=3,
               ncol=7,
               fontsize=18,
               frameon=True)

    fig.savefig(f'./figures/deeppara3_2.png', dpi=300)


    return fig, axes


def plot_para_figure5(data_list, our_methods, figsize=(16, 10)):  # 稍微增加高度
    """
    绘制实验对比图 - 5图倒等腰梯形布局
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    fig = plt.figure(figsize=figsize)

    # 创建 2行 6列 的网格
    # 第一行 3个子图，每个占 2 列: (0,0-1), (0,2-3), (0,4-5)
    # 第二行 2个子图，每个占 2 列，居中则占用: (1,1-2), (1,3-4)
    gs = gridspec.GridSpec(2, 6, figure=fig)

    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.1,
        top=0.88,  # 留出空间给顶部的 legend
        wspace=1.2,  # 增加列间距
        hspace=0.4  # 增加行间距
    )

    # 定义 5 个子图的位置坐标
    ax_positions = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],  # 第一行
        gs[1, 1:3], gs[1, 3:5]  # 第二行居中
    ]

    axes = []
    for pos in ax_positions:
        axes.append(fig.add_subplot(pos))

    # 设置颜色方案 (保持原逻辑)
    mycolor = mycolorclass()
    color_list = [
        mycolor.red, mycolor.green, mycolor.brown, mycolor.blue,
        mycolor.black, mycolor.molvse, mycolor.qianlanse,
        mycolor.pink, mycolor.purple, mycolor.yellow,
    ]

    color_dict = {}
    for i, method in enumerate(our_methods):
        color_dict[method] = color_list[i % len(color_list)]

    # 遍历每个环境 (假设 data_list 长度为 5)
    for env_idx, data in enumerate(data_list):
        if env_idx >= 5: break

        env_name = data.env_name
        ax = axes[env_idx]

        # 设置各环境的 ylim (保持原逻辑)
        if env_name == "Asterix":
            ax.set_ylim(9, 14.5)
        elif env_name == "Breakout":
            ax.set_ylim(10, 20.5)
        elif env_name == "Seaquest":
            ax.set_ylim(5, 16)
        elif env_name == "SpaceInvaders":
            ax.set_ylim(40, 60)
        elif env_name == "Pong":
            ax.set_ylim(-5, 0)

        # 坐标轴格式设置
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                      ['0', '2', '4', '6', '8', '10'])
        ax.grid()

        # 绘制方法曲线
        for method in our_methods:
            if method in data.data_dict:
                x_mean = data.data_dict[method]['x_mean']
                y_mean = data.data_dict[method]['y_mean']
                y_ci = data.data_dict[method]['y_ci']

                ax.plot(x_mean, y_mean,
                        color=color_dict[method],
                        linewidth=1.5,
                        linestyle='-',
                        label=method)
                ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci,
                                facecolor=color_dict[method], alpha=0.3)

        # y轴标签 (第一行最左为0，第二行最左为3)
        if env_idx in [0, 3]:
            ax.set_ylabel('Average Return', fontsize=20)

        # 设置环境名称和索引 (a, b, c...)
        figcharidx = chr(ord('a') + env_idx)
        ax.set_xlabel(f'Steps ($t \; x1e5$)\n({figcharidx}) {env_name}', fontsize=20)

    # 统一提取 Legend
    legend_handles = []
    legend_labels = []
    for method in our_methods:
        label = r"$M$=" + method[3:]
        line, = plt.plot([], [], color=color_dict[method],
                         linewidth=3, label=label)
        legend_handles.append(line)
        legend_labels.append(label)

    # 绘制顶部全局图例
    fig.legend(handles=legend_handles,
               labels=legend_labels,
               handlelength=2.3,
               handletextpad=0.6,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               columnspacing=3,
               ncol=min(len(our_methods), 7),
               fontsize=19,
               frameon=True)

    fig.savefig(f'./figures/deeppara3_2.png', dpi=300)

    return fig, axes

def plot_paper_figure(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    绘制实验对比图

    Parameters:
    -----------
    data_list : list
        实验数据列表，每个元素是ExperimentData对象
    baseline_methods : list
        基线方法名称列表
    our_methods : list
        我们的方法名称列表
    figsize : tuple
        图形大小
    """
    plt.rcParams.update({
        'figure.dpi': 300,
    })

    # 创建图形，2行N列
    fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False)
    plt.subplots_adjust(
        left=0.06,  # 左边距 (默认0.125)
        right=0.99,  # 右边距
        bottom=0.11,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.3,  # 水平间距 (默认0.2)
        hspace=0.35  # 垂直间距 (默认0.2)
    )

    # 设置颜色方案

    color_dict = {
        "OrderDQN": 'c',
        "DQN": 'm',
        "DDQN": 'y',
        "AveragedDQN": '#1f77b4',
        "EnsembleDQN": '#ff7f0e',
        "MaxminDQN": 'purple',
        "WeightedDQN": 'olive',
        "EBDQN": 'lime',
        "M-DDQN0": 'orange',
        "M-DDQN0.2": 'brown',
        "M-DDQN0.4": 'red',
        "M-DDQN0.6": 'blue',
        "M-DDQN0.8": 'green',
        "M-DDQN1": 'pink',

    }

    # 遍历每个环境
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        row = env_idx // 4
        col = env_idx % 4
        ax = axes[row, col]

        #MC env ylim
        if env_name == "MountainCar":
            ax.set_ylim(-400, -50)
        #Puckworld ylim
        if env_name == "PuckWorld":
            ax.set_ylim(-2500, -1000)

        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                   ['0', '2', '4', '6', '8', '10'])
        ax.grid()


        # 找到基线+我们的方法中最好的方法
        # 首先找到我们的方法中最好的（基于最后一个y_mean值）
        best_our_method = None
        best_our_value = -float('inf')

        for method in our_methods:
            if method in data.data_dict:
                y_mean = data.data_dict[method]['y_mean']
                original_method_y_mean_len = len(y_mean)
                start_index = int(original_method_y_mean_len * 0.9)
                table_method_y_mean = np.array(y_mean[start_index:]).mean()

                if table_method_y_mean > best_our_value:
                    best_our_value = table_method_y_mean
                    best_our_method = method

        # 绘制基线方法 + 最好的我们的方法
        # 绘制基线方法
        for method in baseline_methods:
            x_mean = data.data_dict[method]['x_mean']
            y_mean = data.data_dict[method]['y_mean']
            y_ci = data.data_dict[method]['y_ci']

            # 只在第一列的子图添加图例标签
            label = method
            ax.plot(x_mean, y_mean,
                    color=color_dict[method],
                    linewidth=1.5,
                    linestyle='-',  # 基线方法用虚线
                    label=label)
            ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        # 绘制最好的我们的方法
        x_mean = data.data_dict[best_our_method]['x_mean']
        y_mean = data.data_dict[best_our_method]['y_mean']
        y_ci = data.data_dict[best_our_method]['y_ci']
        # 只在第一列的子图添加图例标签
        label = f"OurBest"
        ax.plot(x_mean, y_mean,
                color='red',  # 最好的方法用红色突出
                linewidth=1.5,
                label=label)  # 确保在最上层
        ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)

        if env_idx in [0, 4]:
            ax.set_ylabel('Average Return', fontsize=15)
            #ax.set_ylabel('Average Return (M-DDQNx)', fontsize=10)

        # 设置子图标题
        #ax.set_title(env_name, fontsize=15, pad=10)

        figcharidx = chr(ord('a')+env_idx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Comparsion on {env_name}', fontsize=15)
        ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

    #最后两个图
    rho_envs = ["Asterix", "SpaceInvaders"]
    for env_idx, data in enumerate(data_list):
        env_name = data.env_name
        if env_name not in rho_envs:
            continue
        subfigidx = rho_envs.index(env_name) + 6
        row = subfigidx // 4
        col = subfigidx % 4
        ax = axes[row, col]


        ax.set_xlim((0, 1000000))
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000],
                      ['0', '2', '4', '6', '8', '10'])
        #ax.grid()


        # 所有我们的方法
        for method in our_methods:
            if method[-3:] not in ["0.2", "0.4", "0.6", "0.8", "QN0", "QN1"]:
                continue
            if method in data.data_dict:
                x_mean = data.data_dict[method]['x_mean']
                y_mean = data.data_dict[method]['y_mean']
                y_ci = data.data_dict[method]['y_ci']
                # 设置线宽：最好的方法加粗
                linewidth = 2.5 if method == best_our_method else 1.5

                label = method
                ax.plot(x_mean, y_mean,
                        color=color_dict[method],
                        linewidth=linewidth,
                        label=label)
                ax.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_dict[method], alpha=0.5)



        # 设置轴标签
        figcharidx = chr(ord('a') + subfigidx)
        # 设置轴标签
        ax.set_xlabel(f'Steps(x1e5)\n({figcharidx}) Effect of '+ r"""$\rho$"""+f' on {env_name}', fontsize=15)
        # 添加网格
        ax.grid()
        ax.legend(fontsize=14, loc=[0, 0], handlelength=1, framealpha=0.5)

        # # 自动调整y轴范围，留出一些边距


    # 创建图例
    # 第一行图例（基线方法 + 最好的我们的方法）
    # top_legend_handles = []
    # top_legend_labels = []
    #
    # # 基线方法
    # for method in baseline_methods:
    #     line, = plt.plot([], [], color=color_dict[method],
    #                      linewidth=1.5, linestyle='-', label=method)
    #     top_legend_handles.append(line)
    #     top_legend_labels.append(method)
    #
    # # 最好的我们的方法
    # line_best, = plt.plot([], [], color='red',
    #                       linewidth=2.5, label='OurBest')
    # top_legend_handles.append(line_best)
    # top_legend_labels.append('OurBest')
    #
    # showed_our_methods = [f'M-DDQN{i:.1f}' for i in np.arange(0.2, 1, 0.2)]
    # showed_our_methods += ['M-DDQN0', 'M-DDQN1']
    #
    # for method in showed_our_methods:
    #     line, = plt.plot([], [], color=color_dict[method],
    #                      linewidth=1.5, label=method)
    #     top_legend_handles.append(line)
    #     top_legend_labels.append(method)
    #
    # # 添加第一行图例（居中，在图形上方）
    # fig.legend(handles=top_legend_handles,
    #            labels=top_legend_labels,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 0.99),  # 在图形正上方
    #            ncol=len(top_legend_handles) // 2 if len(top_legend_handles) % 2 == 0 else len(
    #                top_legend_handles) // 2 + 1,
    #            fontsize=15,
    #            frameon=True)

    # 调整布局，为图例留出空间
    #plt.tight_layout(rect=[0, 0, 1, 1])  # 顶部留出10%空间给图例

    return fig, axes

class Plotter(object):
    def __init__(self, cfg):
        cfg.setdefault('ci', None)
        self.x_label = cfg['x_label']
        self.y_label = cfg['y_label']
        self.show = cfg['show']
        self.imgType = cfg['imgType']
        self.ci = cfg['ci']
        self.runs = cfg['runs']
        make_dir('figures/')

    def get_result(self, exp, config_idx, mode):
        '''
        Given exp and config index, get the results
        '''
        total_combination = get_total_combination(exp)
        result_list = []
        for _ in range(self.runs):
            result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
            # If result file exist, read and merge
            result = read_file(result_file)
            if result is not None:
                # Add config index as a column
                result['Config Index'] = config_idx
                result_list.append(result)
            config_idx += total_combination

        # Do symmetric EMA (exponential moving average)
        # Get x's and y's in form of numpy arries
        xs, ys = [], []
        for result in result_list:
            xs.append(result[self.x_label].to_numpy())
            ys.append(result[self.y_label].to_numpy())
        # Do symetric EMA to get new x's and y's
        low = max(x[0] for x in xs)
        high = min(x[-1] for x in xs)
        n = min(len(x) for x in xs)
        for i in range(len(xs)):
            new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
            result_list[i] = result_list[i][:n]
            result_list[i].loc[:, self.x_label] = new_x
            result_list[i].loc[:, self.y_label] = new_y

        ys = []
        for result in result_list:
            ys.append(result[self.y_label].to_numpy())
        # Compute x_mean, y_mean and y_ci
        ys = np.array(ys)
        x_mean = result_list[0][self.x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        if self.ci == 'sd':
            y_ci = np.std(ys, axis=0, ddof=0)
        elif self.ci == 'se':
            y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))

        return x_mean, y_mean, y_ci


def x_format(x, pos):
    # return '$%.1f$x$10^{6}$' % (x/1e6)
    return '%.1f' % (x / 1e6)


cfg = {
    'x_label': 'Step',
    'y_label': 'Average Return',
    'show': False,
    'imgType': 'png',
    'ci': 'se',
    'x_format': None,
    'y_format': None,
    'xlim': {'min': None, 'max': None},
    'ylim': {'min': None, 'max': None},
    'runs': 10,
    'loc': 'lower right'
}


def get_num_combinations_of_dict_before_key(config_dict, target_key_seq):
    '''
    Get # of combinations for configurations in a config dict
    '''
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    num_combinations_of_key = None
    for key, values in config_dict.items():
        if key == target_key_seq[0]:
            if len(target_key_seq) == 1:
                num_combinations_of_key = len(values)
                break
            else:
                num_combinations_of_list, num_combinations_of_key = get_num_combinations_of_list(values, target_key_seq[1:])
                num_combinations_of_dict *= num_combinations_of_list
                if num_combinations_of_key is not None:
                    break


        else:
            num_combinations_of_list, num_combinations_of_key = get_num_combinations_of_list(values, target_key_seq)
            num_combinations_of_dict *= num_combinations_of_list
            if num_combinations_of_key is not None:
                break

    config_dict['num_combinations'] = num_combinations_of_dict
    return num_combinations_of_dict, num_combinations_of_key


def get_num_combinations_of_list(config_list, target_key):
    '''
    Get # of combinations for configurations in a config list
    '''
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    num_combinations_of_key = None
    for value in config_list:
        if type(value) == dict:
            if not ('num_combinations' in value.keys()):
                _, num_combinations_of_key = get_num_combinations_of_dict_before_key(value, target_key)
            num_combinations_of_list += value['num_combinations']
        else:
            num_combinations_of_list += 1
    return num_combinations_of_list, num_combinations_of_key

def cluster_single_exp_idx(exp):
    config_file_path = f'./configs/{exp}.json'
    config = json.load(open(config_file_path, 'r'))
    total_combination = get_total_combination(exp)
    if "_dqn" in exp or "_Vanilladqn" in exp:
        key_seq = ["agent", "name"]
    elif "rho" in exp:
        key_seq = ["agent", "M"]
    elif "Maxmin" in exp or "Averaged" in exp or "Ensemble" in exp:
        key_seq = ["agent", "target_networks_num"]
    elif "Weighted" in exp:
        key_seq = ["agent", "c"]
    elif "MixData" in exp or "Target2MixDataDDQN" in exp:
        key_seq = ["agent", "rho"]
    elif "Target2DDQN" in exp:
        key_seq = ["agent", "sel"]
    elif "AdaODQN" in exp:
        key_seq = ["agent", "m"]
    elif "EBDQN" in exp:
        key_seq = ["agent", "target_networks_num"]
    elif "ActionMultiplex" in exp:
        key_seq = ["agent", "multiplex"]
    elif "ACCDDQN" in exp:
        key_seq = ["agent", "multiplex"]
    else:
        raise NotImplementedError

    num_combinations_before_key, num_combinations_of_key = get_num_combinations_of_dict_before_key(config, key_seq)
    num_combinations_after_key = int(total_combination / (num_combinations_before_key * num_combinations_of_key))
    all_keys_group = []
    for i in range(num_combinations_of_key):
        key_group = []
        row_start_idx = i * num_combinations_after_key

        for j in range(num_combinations_before_key):
            start_idx = row_start_idx + (j*num_combinations_after_key*num_combinations_of_key)
            end_idx = start_idx + num_combinations_after_key
            idxs = range(start_idx, end_idx)
            key_group+=idxs
        all_keys_group.append(key_group)

    all_keys_group = np.array(all_keys_group).T+1 #make idx begin from 1 not 0
    ##N_setting every setting has idx

    result = np.vectorize(lambda x: (exp, x), otypes=[object])(all_keys_group)

    return result

def cluster_exp_idx(exp_list):
    multi_exp_idx_group_list = [] #
    for exp in exp_list:
        exp_idx_group_list = cluster_single_exp_idx(exp) # np: one element correspond to one type env setting
        multi_exp_idx_group_list.append(exp_idx_group_list)
    if len(multi_exp_idx_group_list) == 0:
        return []
    exp_idx_group_list = np.concatenate(multi_exp_idx_group_list, axis=1)
    return exp_idx_group_list




def generate_figure_by_method_metric(methods, metrics, title="env"):


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 200,
    })

    color_list = plt.cm.tab20.colors[:len(methods)]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    all_method_x_mean = metrics["x_mean"]
    all_method_y_mean = metrics["y_mean"]
    all_method_y_ci = metrics["y_ci"]
    for i in range(len(methods)):
        label = methods[i]
        x_mean, y_mean, y_ci = all_method_x_mean[i], all_method_y_mean[i], all_method_y_ci[i]
        first_index_search = np.searchsorted(x_mean, 5e5, side='right')
        # x_mean = x_mean[0:first_index_search]
        # y_mean = y_mean[0:first_index_search]
        # y_ci = y_ci[0:first_index_search]
        plt.plot(x_mean, y_mean, linewidth=1.5, color=color_list[i], label=label)
        if cfg['ci'] in ['se', 'sd']:
            plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_list[i], alpha=0.5)
            # Set x and y axis
    # ax.set_xlabel("Steps (x$10^{6}$)", fontsize=16)
    # ax.set_ylabel('Average Return', fontsize=16, rotation='horizontal')
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel('Average Return', fontsize=16)
    ax.set_ylim(-1250, -1050)
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.title(title, fontsize=16)
    # # Set legend
    ax.legend(loc=cfg['loc'], frameon=False, fontsize=12)
    # Adjust layout automatically
    plt.tight_layout()
    # Save and show

    image_path = f'./figures/{title}.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)

    plt.show()

def create_latex_table(row_names, data_dict, title="table", label="tab:my_table"):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """
    col_names = list(data_dict.keys())

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(col_names) + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白
    header = " & ".join([""] + col_names) + r" \\"
    latex_code.append(header)
    latex_code.append(r"\hline ")

    # 数据行
    for i, row_name in enumerate(row_names):
        row_data = [row_name]
        for col_name in col_names:
            data = data_dict[col_name]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "improve":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)

def table_experiment_comparison(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """


    env_names = []
    for env in data_list:
        env_names.append(env.env_name)

    #row_names = list(data_list[0].data_dict.keys())

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(env_names) + "}")
    latex_code.append(r"\hline ")

    # 表头 - 左上角空白
    header = r"\multirow{2.5}{*}{Algorithm}  &\multicolumn{"+str(len(env_names))+r"}{c}{Averaged reward per episode}"+r" \\"
    latex_code.append(header)
    latex_code.append(r"\cmidrule(r){2-"+str(len(env_names)+1)+r"}")
    header = " & ".join([""]+[env_name for env_name in env_names]) + r" \\"
    latex_code.append(header)
    latex_code.append(r"\midrule")


    # hline = r""
    # for i in range(len(data_list)):
    #     hline += r" \cmidrule(lr){"+str(i*2+2)+"-"+str(i*2+3)+"}"
    # latex_code.append(hline)

    col_names = []
    for env in data_list:
        col_names.append("Reward")
        #col_names.append("Improv")

    ordered_method_nams = [
        "OrderDQN",
    "DQN",
    "DDQN",
    "AveragedDQN",
    "MaxminDQN",
    "WeightedDQN",
    "EBDQN",
    "ACCDDQN",

    #"MIX8",
    "MIX9",
    "MIX10",
    "MIX11",
    "MIX12",
    "MIX13",
    "MIX14",
    "MIX15",
    #"MIX16",
    ]
    method2rowNo_dict = dict(zip(ordered_method_nams, range(len(ordered_method_nams))))

    row_names = ordered_method_nams
    table_data = np.zeros((len(row_names), len(env_names)*1))
    for j, env in enumerate(data_list):
        for _, method in enumerate(row_names):

            env_method_y_mean = env.data_dict[method]["y_mean"]
            original_method_y_mean_len = len(env_method_y_mean)
            start_index = int(original_method_y_mean_len * 0.9)
            table_method_y_mean = np.array(env_method_y_mean[start_index:]).mean()
            i = method2rowNo_dict[method]
            table_data[i][j] = table_method_y_mean


    # for j, env in enumerate(data_list):
    #     reward_col = table_data[:, j * 2]
    #     improve_col = (reward_col - reward_col[0])/np.abs(reward_col[0])
    #     table_data[:,j*2+1] = improve_col
    # 数据行
    for i, row_name in enumerate(ordered_method_nams):
        if "MIX" in row_name:
            row_data = [r"$M$="+row_name[3:]]
        else:
            row_data = [row_name]
        if i==len(baseline_methods):
            latex_code.append(r"\midrule")

        for j, col_name in enumerate(col_names):
            data = table_data[:,j]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "Improv":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\bottomrule ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)

def table_experiment_comparison_RrwImp(data_list, baseline_methods, our_methods, figsize=(16, 8)):
    """
    生成LaTeX三线表代码
    row_names: method names
    data_dict: metric_name:methods_metric_value
    """


    env_names = []
    for env in data_list:
        env_names.append(env.env_name)

    row_names = list(data_list[0].data_dict.keys())

    latex_code = []
    latex_code.append(r"\begin{tabular}{l" + "c" * len(env_names)*2 + "}")
    latex_code.append(r"\toprule")


    # 表头 - 左上角空白
    header = " & ".join([r"\multirow{2.5}{*}{Algorithm}"]+[r"\multicolumn{2}{c}{"+env_name+"}" for env_name in env_names]) + r" \\"
    latex_code.append(header)

    hline = r""
    for i in range(len(data_list)):
        hline += r" \cmidrule(lr){"+str(i*2+2)+"-"+str(i*2+3)+"}"
    latex_code.append(hline)


    col_names = []
    for env in data_list:
        col_names.append("Reward")
        col_names.append("Improv")

    title = " & ".join([""] + col_names) + r" \\"
    latex_code.append(title)

    latex_code.append(r"\midrule")

    ordered_method_nams = [
        "OrderDQN",
    "DQN",
    "DDQN",
    "AveragedDQN",
    "MaxminDQN",
    "WeightedDQN",
    "EBDQN",
    "ACCDDQN",

    "topK2",
    "topK3",
    "topK4",
    "topK5",
    "topK6",
    "topK7",
    "topK8",
    ]
    method2rowNo_dict = dict(zip(ordered_method_nams, range(len(ordered_method_nams))))


    table_data = np.zeros((len(row_names), len(env_names)*2))
    for j, env in enumerate(data_list):
        for _, method in enumerate(row_names):

            env_method_y_mean = env.data_dict[method]["y_mean"]
            original_method_y_mean_len = len(env_method_y_mean)
            start_index = int(original_method_y_mean_len * 0.9)
            table_method_y_mean = np.array(env_method_y_mean[start_index:]).mean()
            i = method2rowNo_dict[method]
            table_data[i][j*2] = table_method_y_mean


    for j, env in enumerate(data_list):
        reward_col = table_data[:, j * 2]
        improve_col = (reward_col - reward_col[0])/np.abs(reward_col[0])
        table_data[:,j*2+1] = improve_col

        baselines_reward_col = table_data[:len(baseline_methods), j * 2]
        sota_reward = np.max(baselines_reward_col)
        table_data[:,j*2+1] = (reward_col - sota_reward)/np.abs(sota_reward)
    # 数据行
    for i, row_name in enumerate(ordered_method_nams):
        row_data = [row_name]
        if i==len(baseline_methods):
            latex_code.append(r"\midrule")

        for j, col_name in enumerate(col_names):
            data = table_data[:,j]
            sorted_idx = np.argsort(data)[::-1]
            if col_name == "Improv":
                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]*100:.2f}}}" + r'\%')  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]*100:.2f}}}" + r'\%')  # 次大值下划线
                else:
                    row_data.append(f"{data[i]*100:.2f}" + r'\%')
            else:

                if i == sorted_idx[0]:
                    row_data.append(fr"\textbf{{{data[i]:.2f}}}")  # 最大值加粗
                elif len(data) > 1 and i == sorted_idx[1]:
                    row_data.append(fr"\underline{{{data[i]:.2f}}}")  # 次大值下划线
                else:
                    row_data.append(f"{data[i]:.2f}")

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\bottomrule ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)

def render_latex_table(table_tex_code, title):

    # 创建完整的LaTeX文档
    print(table_tex_code)
    full_latex_code = table_tex_code.replace("\n", " ")
    #print(full_latex_code)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')

    # 渲染LaTeX表格
    ax.text(0.5, 0.5, full_latex_code, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            horizontalalignment='center')

    image_path = f'./figures/{title}_table.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)
    #plt.tight_layout()
    plt.show()


def reorder_methods(method_names):
    # 分离三种类型的方法
    ddqn_mask = [m == 'DDQN' for m in method_names]
    dqn_mask = [m == 'DQN' for m in method_names]
    other_mask = [m not in ['DQN', 'DDQN'] for m in method_names]

    # 重新排序
    new_order = np.where(ddqn_mask)[0].tolist() + \
                np.where(other_mask)[0].tolist() + \
                np.where(dqn_mask)[0].tolist()
    return new_order


def generate_table_by_method_metric(methods, metrics, title):
    row_names = methods
    Reward_col = []
    all_method_y_mean = metrics["y_mean"]
    for method_y_mean in all_method_y_mean:
        original_method_y_mean_len = len(method_y_mean)
        start_index = int(original_method_y_mean_len*0.9)
        table_method_y_mean = np.array(method_y_mean[start_index:]).mean()
        Reward_col.append(table_method_y_mean)
    Reward_col = np.array(Reward_col).flatten()


    improve_col = (Reward_col - Reward_col[0]) / np.abs(Reward_col[0])

    data_dict = {"Reward": Reward_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict)
    # test_render_latex_table()
    render_latex_table(latex_code, title)



def get_method_metrics_by_exp_idx_gropu(expidxs_list):
    plotter = Plotter(cfg)

    template_exp, template_config_idx = expidxs_list[0]
    template_config_path = f'./logs/{template_exp}/{template_config_idx}/config.json'
    template_config = json.load(open(template_config_path, "r"))
    env = template_config["env"]['name']


    methods = []
    x_mean_metrics = []
    y_mean_metrics = []
    y_ci_metrics = []
    for i in range(len(expidxs_list)):
        exp, config_idx = expidxs_list[i]
        config_path = f'./logs/{exp}/{config_idx}/config.json'
        config = json.load(open(config_path, "r"))
        agent = config['agent']

        if "_dqn" in exp or "Vanilladqn" in exp:
            label = config['agent']['name']
        elif "Vanillarho" in exp:
            K = config['agent']['networks_num']
            M = config['agent']['M']
            label = f'VrhoK{K}M{M}'
            label = f'MIX{M}'
        elif "_rho" in exp:
            agent = config['agent']['name']
            K = config['agent']['networks_num']
            M = config['agent']['M']
            label = f'rhoK{K}M{M}'
        elif "Maxmin" in exp or "Averaged" in exp or "Ensemble" in exp:
            agent = config['agent']['name']
            K = config['agent']['target_networks_num']
            label = f'{agent}{K}'
            label = f'{agent}'
        elif "Weighted" in exp:
            agent = config['agent']['name']
            c = config['agent']['c']
            label = f'{agent}{c}'
            label = f'WeightedDQN'
        elif "MixData" in exp or "Target2MixDataDDQN" in exp:
            agent = config['agent']['name']
            rho = config['agent']['rho']
            label = f'{agent}{rho}'
            label = f'M-DDQN{rho}'
        elif "Target2DDQN" in exp:
            agent = config['agent']['name']
            sel = config['agent']['sel']
            label = f'{agent}{sel}sel'
        elif "AdaODQN" in exp:
            agent = config['agent']['name']
            M = config['agent']['M']
            m = config['agent']['m']
            # if int(m) != 1:
            #     continue
            label = f'{agent}M{M}m{m}'
            label = f'OrderDQN'
        elif "EBDQN" in exp:
            agent = config['agent']['name']
            K = config['agent']['target_networks_num']
            label = f'{agent}{K}'
            label = f'EBDQN'
        elif "ActionMultiplex" in exp:
            agent = config['agent']['name']
            K = config['agent']['multiplex']
            label = f'topK{K}'
        elif "ACC" in exp:
            agent = config['agent']['name']
            K = config['agent']['multiplex']
            label = f'ACCDDQN'
        else:
            raise NotImplementedError
        methods.append(label)
        print(f'[{exp}]: Plot Train results: {config_idx}')
        x_mean, y_mean, y_ci = plotter.get_result(exp, config_idx, 'Train')
        x_mean_metrics.append(x_mean)
        y_mean_metrics.append(y_mean)
        y_ci_metrics.append(y_ci)

    new_order = reorder_methods(methods)

    methods = [methods[i] for i in new_order]
    metric_dict = {
        "x_mean": [x_mean_metrics[i] for i in new_order],
        "y_mean": [y_mean_metrics[i] for i in new_order],
        "y_ci": [y_ci_metrics[i] for i in new_order],
    }

    return methods, metric_dict

def getexp_idxenv(exp, confgi_idx):
    config_path = f'./logs/{exp}/{confgi_idx}/config.json'
    config = json.load(open(config_path, "r"))
    env = config['env']['name']
    if "copter" in env:
        env = "Pixelcopter"
    elif "Asterix" in env:
        env = "Asterix"
    elif "Space" in env:
        env = "SpaceInvaders"
    elif "Seaquest" in env:
        env = "Seaquest"
    elif "MountainCar" in env:
        env = "MountainCar"
    elif "PuckWorld" in env:
        env = "PuckWorld"
    elif "Breakout" in env:
        env = "Breakout"
    elif "NChain" in env:
        env = "NChain"
    elif "CartPole" in env:
        env = "CartPole"
    elif "Pong" in env:
        env = "Pong"
    elif "FlappyBird" in env:
        env = "FlappyBird"
    elif "Acrobot" in env:
        env = "Acrobot"
    # 'LunarLander',
    # 'Catcher',
    elif "LunarLander" in env:
        env = "LunarLander"
    elif "Cat" in env:
        env = "Catcher"
    else:
        raise NotImplementedError
    return env

def learning_curve(exp_list, runs=1):
    cfg['runs'] = runs

    exp_idx_group_list = cluster_exp_idx(exp_list) #one element correspond to one type env setting

    for exp_idx_group in exp_idx_group_list:
        exp, idx = exp_idx_group[0]
        env = getexp_idxenv(exp, idx)
        methods, metrics = get_method_metrics_by_exp_idx_gropu(exp_idx_group)

        # new_methods = []
        # new_metrics = {
        #     "x_mean": [],
        #     "y_mean": [],
        #     "y_ci": [],
        # }
        # for i in range(len(methods)):
        #     if "MixData" in methods[i]:
        #         rho = float(methods[i].split("DQN")[-1])
        #         if rho != 0.4:
        #             continue
        #
        #     # if methods[i] == "DDQN":
        #     #     continue
        #
        #     new_methods.append(methods[i])
        #     new_metrics["x_mean"].append(metrics["x_mean"][i])
        #     new_metrics["y_mean"].append(metrics["y_mean"][i])
        #     new_metrics["y_ci"].append(metrics["y_ci"][i])
        # methods = new_methods
        # metrics = new_metrics

        # generate_figure_by_method_metric(methods, metrics, title=env)
        # generate_table_by_method_metric(methods, metrics, title=env)

        data_dict = {}
        for i in range(len(methods)):
            methodname = methods[i]
            baseline_x_mean = metrics['x_mean'][i]
            baseline_y_mean = metrics['y_mean'][i]
            baseline_y_ci = metrics['y_ci'][i]
            data_dict[methodname] = {
                'x_mean': baseline_x_mean,
                'y_mean': baseline_y_mean,
                'y_ci': baseline_y_ci,
            }
        env1 = ExperimentData(env, data_dict)

        return env1

if __name__ == "__main__":


    envlist = [
        #"copter",
        "Asterix",
        "breakout",
        "seaquestv1",
        #"PuckWorld",
        "SpaceInvadersv1",
        'Pong',
        # 'NChain',
        # 'mc',
        # 'FlappyBird',
        # 'acrobot',
        # 'CartPole',
        #'LunarLander',
        #'Catcher',

    ]
    # config filename
    methodslist = [
        "dqn",
        "Weighted",
        "Maxmin",
        "Averaged",
        "AdaODQN",
        "EBDQN",
        'Vanillarhodqn',
        'ACCDDQN',
    ]

    exp_list = []
    for env in envlist:
        envcfglist = []
        for method in methodslist:
            config_file = f'MERL_{env}_{method}'
            envcfglist.append(config_file)
        exp_list.append(envcfglist)

    #data dict method name
    baseline_methods = ['AveragedDQN', 'WeightedDQN', "DDQN", "DQN", "OrderDQN", "EBDQN", "MaxminDQN","ACCDDQN"]
    #baseline_methods = ['AveragedDQN', 'WeightedDQN', "DDQN", "DQN",  "MaxminDQN"]
    baseline_methods = ["DQN",
                        "MaxminDQN",

                        "DDQN",
                        "EBDQN",

                        'AveragedDQN',
                        "ACCDDQN",

                        'WeightedDQN',
                        "OrderDQN",]
    # baseline_methods = [#"DQN",
    #                     "MaxminDQN",
    #
    #                     #"DDQN",
    #                     "EBDQN",
    #
    #                     'AveragedDQN',
    #                     #"ACCDDQN",
    #
    #                     #'WeightedDQN',
    #                     "OrderDQN", ]

    our_methods = [f'MIX{i:d}' for i in range(9, 16)]
    #our_methods = [f'topK{i:d}' for i in [10,20,40,80,160]]

    data_list = []
    for exp in exp_list:
        expobj = learning_curve(exp, runs=20)
        data_list.append(expobj)



    # 绘图


    # fig, axes = plot_baseline_figure(
    #     data_list=data_list,
    #     baseline_methods=baseline_methods,
    #     our_methods=our_methods,
    #     figsize=(15,12)
    # )
    # plt.show()
    fig, axes = plot_baseline_figure5(
        data_list=data_list,
        baseline_methods=baseline_methods,
        our_methods=our_methods,
        figsize=(15, 12)
    )
    plt.show()

    plot_para_figure5(
        data_list=data_list,
        our_methods=our_methods,
        figsize=(15, 12)
    )
    plt.show()



    plot_para_figure(
        data_list=data_list,
        our_methods=our_methods,
        figsize=(15, 12)
    )
    plt.show()

    table_latex = table_experiment_comparison(
        data_list=data_list,
        baseline_methods=baseline_methods,
        our_methods=our_methods,
        figsize=(16, 8)
    )
    print(table_latex)

    # table_latex = table_experiment_comparison_RrwImp(
    #     data_list=data_list,
    #     baseline_methods=baseline_methods,
    #     our_methods=our_methods,
    #     figsize=(16, 8)
    # )
    # print(table_latex)
    # render_latex_table(table_latex, title='Learning Curve')






    # 可选：保存图形
    # fig.savefig('experiment_results.png', dpi=300, bbox_inches='tight')