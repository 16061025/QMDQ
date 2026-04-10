import os
import re

import gym
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast
import numpy as np
from Envs.makeEnv import makeEnv


def figureQ_bar(multiarm, labels, title="Q", V_star=0, last_percent=0.1):
    """
    生成柱状图，每个柱子是算法最后x%数据的均值

    Parameters:
    multiarm: 多个算法的结果数据列表
    labels: 算法标签列表
    title: 图片标题
    V_star: 参考水平线值
    last_percent: 使用最后多少比例的数据计算均值 (0-1之间)
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    plt.rcParams.update({
        'figure.dpi': 150,
    })
    fig = plt.figure(figsize=(10, 6))
    rect1 = [0.15, 0.15, 0.8, 0.8]
    ax1 = plt.axes(rect1)

    # 计算每个算法最后x%数据的均值
    means = []
    for data in multiarm:
        n_last = int(len(data) * last_percent)
        if n_last < 1:  # 确保至少有一个数据点
            n_last = 1
        last_data = data[-n_last:]
        means.append(np.mean(last_data))

    # 创建柱状图
    x_pos = np.arange(len(labels))
    bars = ax1.bar(x_pos, means, alpha=0.7, edgecolor='black')

    # 添加参考线
    ax1.axhline(y=V_star, color='red', linestyle='--', linewidth=2, label=f'Reference: {V_star}')

    # 在柱子上方显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel(r'Average Q-value (last {}%)'.format(int(last_percent * 100)), fontsize=15)
    ax1.set_xlabel('Algorithms', fontsize=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax1.tick_params(labelsize=12)
    yy = MultipleLocator(1)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid(axis='y', alpha=0.3)
    #ax1.legend(fontsize=12, loc="upper right")

    plt.title(f"Average Max Q-value (last {int(last_percent * 100)}%) - {title}")
    plt.tight_layout()
    plt.savefig(f"./Result/{title}MaxQ_bar.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()

    return means

def figureQ(multiarm, labels, title="Q", V_star=0):
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    plt.rcParams.update({
        'figure.dpi': 150,
    })
    fig, ax1 = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(
        left=0.13,  # 左边距 (默认0.125)
        right=0.985,  # 右边距
        bottom=0.11,  # 底边距 (默认0.11)
        top=0.99,  # 顶边距
        wspace=0.2,  # 水平间距 (默认0.2)
        hspace=0.32  # 垂直间距 (默认0.2)
    )


    x_value = [i for i in range(len(multiarm[0]))]
    j=0
    colors = plt.cm.tab20(np.linspace(0, 1, len(multiarm)))
    cmap = plt.cm.gist_rainbow

    # 调整范围：0.0=红，~0.6=蓝，1.0=紫
    positions = np.linspace(0, 0.7, len(multiarm))
    colors = cmap(positions)[::-1]
    #colors = plt.cm.RdYlBu(np.linspace(0, 1, len(multiarm)-2))[::-1]
    for i in range(len(multiarm)):
        if "Single" in labels[i]:
            ax1.plot(x_value, multiarm[i], label=labels[i], linewidth=3, color="red", linestyle="-")
        elif "Double" in labels[i]:
            ax1.plot(x_value, multiarm[i], label=labels[i], linewidth=3, color="blue", linestyle="-")
        elif "EBQL" in labels[i]:
            ax1.plot(x_value, multiarm[i], label=labels[i], linewidth=3, color="green", linestyle="-")
        elif "Averaged" in labels[i]:
            ax1.plot(x_value, multiarm[i], label=labels[i], linewidth=3, color="black", linestyle="-")
        else:
            ax1.plot(x_value, multiarm[i], label=labels[i], color = colors[j])
            j = j+1

    ax1.axhline(y=V_star, color='black', linewidth=3 , linestyle='--', label="Unbiased")
    ax1.set_yticks([V_star])


    ax1.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax1.set_xlabel(r'Number of actions $(t x 100)$', fontsize=15)
    ax1.set_xlim(0, len(multiarm[0]))
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-10, 55)

    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    #ax1.xaxis.get_offset_text().set_fontsize(20)
    yy = MultipleLocator(20)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()

    ax1.legend(fontsize=12, loc=(0.63,0.7), handlelength=2, framealpha=0.5)
    # fig.legend(loc='upper center',
    #            bbox_to_anchor=(0.5, 1),
    #            ncol=3,
    #            columnspacing=0.71,
    #            handlelength=2.3,
    #            fontsize=12,
    #            frameon=True,
    #            fancybox=True,
    #            shadow=False,
    #            borderpad=0.8)
    #plt.title("MaxQ"+title)
    plt.savefig(f"../Result/example.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()

def figureExample(multiarm, labels, title="Q", V_star=0):
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    plt.rcParams.update({
        'figure.dpi': 150,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(
        left=0.055,  # 左边距
        right=0.99,  # 右边距
        bottom=0.21,  # 底边距（为图例留出空间）
        top=0.975,  # 顶边距
        wspace=0.25,  # 子图水平间距
        hspace=0.32  # 垂直间距
    )

    x_value = [i for i in range(len(multiarm[0]))]
    ax1 = axes[0]
    for i in range(len(multiarm)):
        if "Single" in labels[i]:
            ax1.plot(x_value, multiarm[i], linewidth=3, color="purple", linestyle="-")
        elif "Averaged" in labels[i]:
            ax1.plot(x_value, multiarm[i],  linewidth=3, color="black", linestyle="-")

    ax1.axhline(y=V_star, color='black', linewidth=3, linestyle='--')
    ax1.set_yticks([V_star])

    ax1.set_ylabel(r'Maximum Q-value', fontsize=20)
    ax1.set_xlabel(f'Steps $(t x 100)$\n(A) Coupling Methods', fontsize=20)
    ax1.set_xlim(0, len(multiarm[0]))
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-5, 40)

    ax1.tick_params(labelsize=20)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    # ax1.xaxis.get_offset_text().set_fontsize(20)
    yy = MultipleLocator(20)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    ax1.plot([], [], linewidth=3, label="Q", color="purple", linestyle="-")
    ax1.plot([], [], linewidth=3, label="Averaged Q", color="black", linestyle="-")
    ax1.plot([], [], linewidth=3, label="Unbiased", color="black", linestyle="--")

    ax1.legend(loc='upper center',
               bbox_to_anchor=(0.64, 0.999),
               ncol=1,
               columnspacing=2.5,
               handlelength=1.5,
               fontsize=20,
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=0.3)



    ax2 = axes[1]
    for i in range(len(multiarm)):
        if "Double" in labels[i]:
            ax2.plot(x_value, multiarm[i], linewidth=3, color="blue", linestyle="-")
        elif "EBQL" in labels[i]:
            ax2.plot(x_value, multiarm[i], linewidth=3, color="green", linestyle="-")

    ax2.axhline(y=V_star, color='black', linewidth=3, linestyle='--')
    ax2.set_yticks([V_star])

    #ax2.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax2.set_xlabel(f'Steps $(t x 100)$\n(B) Decoupling Methods', fontsize=20)
    ax2.set_xlim(0, len(multiarm[0]))
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-30, 5)

    ax2.tick_params(labelsize=20)
    xx = MultipleLocator(40)
    ax2.xaxis.set_major_locator(xx)
    # ax2.xaxis.get_offset_text().set_fontsize(20)
    yy = MultipleLocator(20)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.grid()

    ax2.plot([], [], linewidth=3, label="Double Q", color="blue", linestyle="-")
    ax2.plot([], [], linewidth=3, label="EBQL", color="green", linestyle="-")
    #ax2.plot([], [], linewidth=3, label="Unbiased", color="black", linestyle="--")

    ax2.legend(loc='upper center',
               bbox_to_anchor=(0.64, 0.999),
               ncol=1,
               columnspacing=2.5,
               handlelength=1.5,
               fontsize=20,
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=0.3)

    ax3 = axes[2]
    for i in range(len(multiarm)):
        # if "Single" in labels[i]:
        #     ax3.plot(x_value, multiarm[i], linewidth=3, color="purple", linestyle="-")
        # if "Double" in labels[i]:
        #     ax3.plot(x_value, multiarm[i], linewidth=3, color="blue", linestyle="-")
        if "rho" in labels[i]:
            aaa = labels[i][7:]
            if aaa == "10":
                ax3.plot(x_value, multiarm[i], linewidth=3, color="red", linestyle="-")
            elif aaa == "11":
                ax3.plot(x_value, multiarm[i], linewidth=3, color="orange", linestyle="-")
            elif aaa == "9":
                ax3.plot(x_value, multiarm[i], linewidth=3, color="pink", linestyle="-")

    ax3.axhline(y=V_star, color='black', linewidth=3, linestyle='--')
    ax3.set_yticks([V_star])

    #ax3.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax3.set_xlabel(f'Steps $(t x 100)$\n(C) Semi-decoupling Method', fontsize=20)
    ax3.set_xlim(0, len(multiarm[0]))
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-30, 40)

    ax3.tick_params(labelsize=20)
    xx = MultipleLocator(40)
    ax3.xaxis.set_major_locator(xx)
    # ax3.xaxis.get_offset_text().set_fontsize(20)
    yy = MultipleLocator(20)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_offset_text().set_fontsize(20)
    ax3.grid()

    #ax3.plot([], [], linewidth=3, label="Q", color="purple", linestyle="-")
    ax3.plot([], [], linewidth=3, label=r"QMDQ $M$=11", color="orange", linestyle="-")
    ax3.plot([], [], linewidth=3, label=r"QMDQ $M$=10", color="red", linestyle="-")
    ax3.plot([], [], linewidth=3, label=r"QMDQ $M$=9", color="pink", linestyle="-")
    #ax3.plot([], [], linewidth=3, label="Double Q", color="blue", linestyle="-")
    #ax3.plot([], [], linewidth=3, label="Unbiased", color="black", linestyle="--")

    ax3.legend(loc='upper center',
               bbox_to_anchor=(0.64, 0.999),
               ncol=1,
               columnspacing=2.5,
               handlelength=1.5,
               fontsize=20,
               frameon=True,
               fancybox=True,
               shadow=False,
               borderpad=0.3)

    # methods = ["Q", "Averaged Q", "Double Q", "EBQL", "DMDQ"]
    # method2color = {"Q":"red",
    #                 "Averaged Q":"black",
    #                 "Double Q":"blue",
    #                 "EBQL":"green",
    #                 "DMDQ":"purple"
    #                 }
    # for method in methods:
    #     ax1.plot([], [], label=method, linewidth=3, color=method2color[method], linestyle="-")
    # ax1.axhline(y=V_star, color='black', linewidth=3, linestyle='--', label="Unbiased")
    #
    # fig.legend(loc='upper center',
    #            bbox_to_anchor=(0.5, 1),
    #            ncol=6,
    #            columnspacing=2.5,
    #            handlelength=2.3,
    #            fontsize=20,
    #            frameon=True,
    #            fancybox=True,
    #            shadow=False,
    #            borderpad=0.3)

    plt.savefig(f"../Result/example.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()


def figureR(multiarm, labels, title="R", smooth=False, window_size= 5):
    if smooth:
        for i in range(len(multiarm)):
            multiarm[i] = moving_average(multiarm[i], window_size=window_size)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    plt.rcParams.update({
        'figure.dpi': 150,
    })
    fig = plt.figure(figsize=(6, 7))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.15, 0.15, 0.8, 0.6]
    ax1 = plt.axes(rect1)


    x_value = [i for i in range(len(multiarm[0]))]

    for i in range(len(multiarm)):
        ax1.plot(x_value, multiarm[i], label=labels[i])
    # if "grid" in title:
    #     ax1.axhline(y=0.36, color='black', linestyle='--')
    # else:
    #     ax1.axhline(y=0, color='black', linestyle='--')

    ax1.set_ylabel(r'$r$ per step', fontsize=15)
    ax1.set_xlabel(r'Number of actions $(i x 100)$', fontsize=15)
    ax1.set_xlim(0, len(multiarm[0]))
    #ax1.set_xlim(0, 50)
    #ax1.set_ylim(-1, 0)

    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(10)
    ax1.xaxis.set_major_locator(xx)
    #yy = MultipleLocator(100)
    #ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    #ax1.legend(fontsize=10, loc="lower right", handlelength=1)
    plt.legend(
        loc='upper center',  # 顶部居中
        bbox_to_anchor=(0.5, 1.35),  # 位置调整（水平居中，垂直在图上方的1.15倍高度处）
        ncol=5,  # 每行3列
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=True,
        title_fontsize=12
    )
    plt.title("Avgr"+title)
    #plt.savefig(f"./Result/{title}MaxQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()

def figurePrglobal(multiarm, labels, title="Pr", smooth=False):
    if smooth:
        for i in range(len(multiarm)):
            multiarm[i] = moving_average(multiarm[i], window_size=5)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    plt.rcParams.update({
        'figure.dpi': 150,
    })
    fig = plt.figure(figsize=(6, 5))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.15, 0.15, 0.8, 0.8]
    ax1 = plt.axes(rect1)

    x_value = [i for i in range(len(multiarm[0]))]

    for i in range(len(multiarm)):
        ax1.plot(x_value, multiarm[i], label=labels[i])

    ax1.set_ylabel(r'Pr($s_a=right$)', fontsize=15)
    ax1.set_xlabel(r'Number of actions $(i x 100)$', fontsize=15)
    ax1.set_xlim(0, len(multiarm[0]))

    ax1.set_ylim(0, 1)

    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(10)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.2)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=10, loc="lower right", handlelength=1)
    plt.title(r'Pr($s_a=right$)'+title)
    #plt.savefig(f"./Result/{title}PrsaGlobal.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()

def figurePrlocal(multiarm, labels, title="Q"):
    fig = plt.figure(figsize=(6, 5))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.15, 0.15, 0.8, 0.8]
    ax1 = plt.axes(rect1)

    x_value = [i for i in range(len(multiarm[0]))]

    for i in range(len(multiarm)):
        ax1.plot(x_value, multiarm[i], label=labels[i])

    ax1.set_ylabel(r'Pr($s_a=right$)', fontsize=15)
    ax1.set_xlabel(r'Number of actions $(i x 100)$', fontsize=15)

    #ax1.set_xlim(1, 5)

    ax1.set_ylim(0, 0.6)

    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(10)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.15)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=10, loc="lower right", handlelength=1)
    plt.title(r'Pr($s_a=right$)'+title)
    #plt.savefig(f"./Result/{title}PrsaLocal.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()


def create_latex_table(row_names, data_dict, title="table"):
    """
    生成LaTeX三线表代码
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
                afterfix=r"\%"
            else:
                afterfix=""
            if i == sorted_idx[0]:
                row_data.append(fr"\textbf{{{data[i]:.3f}}}"+afterfix)  # 最大值加粗
            elif len(data) > 1 and i == sorted_idx[1]:
                row_data.append(fr"\underline{{{data[i]:.3f}}}"+afterfix)  # 次大值下划线
            else:
                row_data.append(f"{data[i]:.3f}"+afterfix)

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    return "\n".join(latex_code)



def render_latex_table(table_tex_code, title):

    # 创建完整的LaTeX文档
    print(table_tex_code)
    full_latex_code = table_tex_code.replace("\n", " ")


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        'figure.dpi': 200,
    })

    fig, ax = plt.subplots(figsize=(3.5, 8))
    ax.axis('off')

    # 渲染LaTeX表格
    ax.text(0.5, 0.5, full_latex_code, transform=ax.transAxes,
            fontsize=10, verticalalignment='center',
            horizontalalignment='center')

    #plt.tight_layout()
    plt.title(title, fontsize=10)
    plt.show()



def tableQ(multiarm, labels, title="Q", V_star=0):

    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]

    row_names = labels


    multiarm = np.array(multiarm).flatten()
    Q_col = multiarm
    absMaxQ = np.abs(Q_col-V_star)
    improve_col = (absMaxQ[0] - absMaxQ) / absMaxQ[0]
    data_dict = {"MaxQ":Q_col,
                 "abserrMaxQ": absMaxQ,
                 "improve": improve_col}



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


            if col_name == "MaxQ":
                row_data.append(f"{data[i]:.2f}")
                continue

            if "abs" in col_name:
                sorted_idx = np.argsort(data)
            else:
                sorted_idx = np.argsort(data)[::-1]


            if col_name == "improve":
                afterfix = r"\%"
            else:
                afterfix = ""



            if i == sorted_idx[0]:
                row_data.append(fr"\textbf{{{data[i]:.2f}}}" + afterfix)  # 最值加粗
            elif len(data) > 1 and i == sorted_idx[1]:
                row_data.append(fr"\underline{{{data[i]:.2f}}}" + afterfix)  # 次最值下划线
            else:
                row_data.append(f"{data[i]:.2f}" + afterfix)

        latex_code.append(" & ".join(row_data) + r" \\")

    latex_code.append(r"\hline ")
    latex_code.append(r"\end{tabular}")

    latex_code = "\n".join(latex_code)

    #latex_code = create_latex_table(row_names, data_dict, title=title)
    #test_render_latex_table()
    render_latex_table(latex_code, title)

def tablePrglobal(multiarm, labels, title="Pr"):
    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]

    row_names = labels

    multiarm = np.array(multiarm).flatten()
    Pr_col = multiarm
    improve_col = (multiarm - multiarm[0]) / multiarm[0]
    data_dict = {"Pr": Pr_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict, title=title)
    # test_render_latex_table()
    render_latex_table(latex_code, title)

def tableReward(multiarm, labels, title="R", smooth=False, window_size=5, last_percent=0.1):



    if smooth:
        for i in range(len(multiarm)):
            multiarm[i] = moving_average(multiarm[i], window_size=window_size)

    row_names = labels

    for i in range(len(multiarm)):
        multiarm[i] = multiarm[i][-1]
    multiarm = np.array(multiarm).flatten()
    R_col = multiarm
    improve_col = (multiarm - multiarm[0]) / np.abs(multiarm[0])
    data_dict = {"Reward": R_col,
                 "improve": improve_col}

    latex_code = create_latex_table(row_names, data_dict)
    # test_render_latex_table()
    render_latex_table(latex_code, title)

def moving_average(data, window_size):
    """移动平均平滑"""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')



def get_Q_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-5][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S

def get_Pr_sa(dir):
    log = open(dir, 'r').readlines()
    Pr_sa = log[-3][:]
    Pr_sa = ast.literal_eval(Pr_sa)
    return Pr_sa

def get_r_perstep(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-1][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S

def cfg2graph(cfgdir, cfgname, algorithms, Klist):
    data_len = 100

    pngname = cfgname



    Result_path = os.path.join(cfgdir, cfgname)
    multiarmQ = []
    multiarmPr = []

    multiarmR = []
    labels = []
    for algorithm in os.listdir(Result_path):
        if algorithm not in algorithms:
            continue
        algorithm_dir = os.path.join(Result_path, algorithm)
        if not os.path.isdir(algorithm_dir):
            continue


        if algorithm == "rhoFixOverQ":
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+K(\d+)M(\d+)(.{3})_update\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    K = int(match.group(1))
                    M = int(match.group(2))
                    update_Q = match.group(3)
                    if K not in Klist:# or K != 2*M:
                        continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    labels.append(f'{algorithm[0:3]}K{K}M{M}')
        elif algorithm in ['mixDQ', "mixAvgDQ"]:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+mr(\d+\.\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    pQ = float(match.group(1))
                    hour = int(match.group(2))
                    # if pQ != 0.4:
                    #     continue
                    # if hour != 3:
                    #     continue

                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])
                    if algorithm == "mixAvgDQ":
                        labels.append(f'mixAvg{pQ:.2f}')
                    elif algorithm == "mixDQ":
                        labels.append(f'mix{pQ:.2f}')
        elif algorithm in ['AMultiplexQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+mr(\d+\.\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    pQ = float(match.group(1))
                    hour = int(match.group(2))
                    # if pQ != 0.4:
                    #     continue
                    # if hour != 3:
                    #     continue

                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    labels.append(f'mix{pQ:.2f}')
        elif algorithm in ['AveragedQ', "EnsembleQ", "EBQL", "KQ", "MaxminQ"]:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+K(\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字

                    K = float(match.group(1))
                    if K not in Klist:
                        continue
                    if algorithm == "AveragedQ":
                        if K != 2:
                            continue
                    elif algorithm == "EBQL":
                        if K != 3:
                            continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    #labels.append(f'{algorithm}K{K}')
                    labels.append(f'{algorithm}')
        elif algorithm in ['WeightedQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+c(\d+\.\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    c = float(match.group(1))

                    # if c != 10:
                    #     continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    labels.append(f'{algorithm}c{c}')

        elif algorithm in ['AdaOQ']:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Qenv.+M(\d+)m(\d+)\s+\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d{2}\.\d{2}$'
                match = re.search(pattern, filename)

                if match:
                    # 提取S后的数字和E后的数字
                    M = int(match.group(1))
                    m = int(match.group(2))
                    if m==1:
                        continue
                    # if c != 10:
                    #     continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])

                    labels.append(f'{algorithm}M{M}m{m}')
        elif algorithm in ['SingleQ', "DoubleQ"]:
            for filename in os.listdir(algorithm_dir):
                pattern = r'log\.Q.*(?<!\.(npz|txt))$'
                match = re.search(pattern, filename)
                if match:
                    # if int(match.group(1)) < 15:
                    #    continue
                    y_value = get_Q_value(os.path.join(algorithm_dir, filename))
                    multiarmQ.append(y_value[0:data_len])
                    y_value = get_Pr_sa(os.path.join(algorithm_dir, filename))
                    multiarmPr.append(y_value[0:data_len])
                    y_value = get_r_perstep(os.path.join(algorithm_dir, filename))
                    multiarmR.append(y_value[0:data_len])
                    labels.append(algorithm)

        else:
            raise NotImplementedError
    V_star = 0
    if (cfgname[:4]) =="grid":
        cfgname_pattern = r'.+gamma(.+)Nrow(.+)r(.+)r(.+)gr(.+)gr(.+)adplr.+'
        match = re.search(cfgname_pattern, cfgname)
        gamma, Nrow, r1, r2, gr1, gr2 = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5)), float(match.group(6))
        V_star = calculateV_star(r1, r2, gr1, gr2, Nrow, gamma)

    smooth = False
    window_size = 5
    # idx = labels.index('DoubleQ')
    # doubleq_label = labels.pop(idx)
    # labels.insert(0, doubleq_label)
    #
    # doubleq_Q = multiarmQ.pop(idx)
    # multiarmQ.insert(0, doubleq_Q)
    # doubleq_R = multiarmR.pop(idx)
    # multiarmR.insert(0, doubleq_R)
    # doubleq_Pr = multiarmPr.pop(idx)
    # multiarmPr.insert(0, doubleq_Pr)

    if "grid" in cfgname:
        figureQ(multiarmQ, labels, title=pngname, V_star=V_star)
        #
        figureR(multiarmR, labels, title=pngname, smooth=smooth, window_size=window_size)
        #
        tableQ(multiarmQ, labels, title=pngname, V_star=V_star)
        tableReward(multiarmR, labels, title=pngname, smooth=smooth, window_size=window_size)

    elif "right" in cfgname or "ABC" in cfgname:
        figureQ(multiarmQ, labels, title=pngname, V_star=V_star)
        #figurePrglobal(multiarmPr, labels, title=pngname, smooth=False)
        #tableQ(multiarmQ, labels, title=pngname, V_star=V_star)
        #tablePrglobal(multiarmPr, labels, title=pngname)

    elif "multi" in cfgname:
        V_star = 0
        #figureQ(multiarmQ, labels, title=pngname, V_star=V_star)
        figureExample(multiarmQ, labels, title=pngname, V_star=V_star)



def calculateV_star(r1, r2, gr1, gr2, nrow, gamma):
    Expect_step = (r1+r2)/2
    Expect_goal = (gr1+gr2)/2
    total_step = int(2*(nrow-1)+1)
    V_star = (Expect_goal + Expect_step*(total_step-1))/total_step

    R = [Expect_goal] + [Expect_step for i in range(total_step-1)]
    V_star = 0
    for step in range(total_step):
        V_star = R[step] + gamma*V_star
    return V_star

if __name__ == "__main__":


    cfglist = [


        #"rightenvlr0.8gamma0.95sm-0.1sv1.0om0.0ov1.0adplrTrue",


        "multiarmenvlr0.8gamma0.95Na10as10.0Qm0.0Qs0.01",

    ]


    cfgdir = os.path.join("../ResultExample")
    for cfgname in os.listdir(cfgdir):
        if cfgname not in cfglist:
            continue
        kcfg = [[8]]
        kcfg = [[4],[8],[16]]
        kcfg = [[1,2,3,4,5,6,7,8]]
        kcfg = [[2,3,16]]
        for Klist in kcfg:

            algorithmlist = ["EBQL", "AveragedQ", "SingleQ", "DoubleQ", "rhoFixOverQ"]
            cfg2graph(cfgdir, cfgname, algorithmlist, Klist)

