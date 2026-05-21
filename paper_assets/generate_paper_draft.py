#!/usr/bin/env python3
"""Generate an agent-oriented paper draft with figures for the pig counting project."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "doc"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MD_PATH = OUT_DIR / "pig_counting_agent_paper_final.md"
DOCX_PATH = OUT_DIR / "pig_counting_agent_paper_final.docx"

GROUP4_CSV = ROOT / "output" / "batch_rerun_group4" / "batch_rerun_group4_results.csv"
GROUP5_CSV = ROOT / "output" / "batch_rerun_group5" / "batch_rerun_group5_results.csv"
GROUP6_CSV = ROOT / "output" / "batch_rerun_group6" / "batch_rerun_group6_results.csv"


TITLE_CN = "面向猪只通行计数 Agent 封装的边缘视觉系统研究"
TITLE_EN = "Research on Agent-oriented Encapsulation of an Edge-vision Pig Passage Counting System"
AUTHOR_LINE = "作者：待填写    单位：待填写    指导教师：待填写"

ABSTRACT_CN = (
    "针对猪场通道场景中人工点数效率低、边缘设备长期运行稳定性不足以及少量争议样本难以"
    "统一评价口径的问题，本文围绕一套已完成部署验证的猪只计数项目，提出一种面向 Agent "
    "封装的边缘视觉系统方法。本文首先保留 YOLOv8 检测、ByteTrack 跟踪和三线双向计数"
    "构成的视觉核心，将其作为 Agent 的感知执行内核；随后在其外部增设状态记忆、策略判断、"
    "动作执行、异常审计和人工复核通道，使系统从单纯“能计数”的脚本集合提升为“可自治运行、"
    "可人机协同复核、可论文化表达”的边缘智能代理。结合现有第四、五、六组实验结果，本文"
    "对带标注视频、长视频和高计数场景进行归纳分析：第四组 18 个带标注视频在文件名标签口径下"
    "获得 88.89% 的 exact-match 和 0.333 的平均绝对误差；在将 17-37头.mp4 的真实值经人工"
    "复核修正为 31 后，exact-match 保持为 88.89%，平均绝对误差下降为 0.111，18 个样本的"
    "累计预测总数与修正后真实值总数同为 648。对于长视频与高计数场景，第五组 long_video.mp4"
    " 得到 total_line 2187、valid_traj 2193；第六组三个视频分别得到 2184/2188、568/565"
    " 和 979/975 的统计结果。"
    "基于此，本文进一步讨论如何将现有系统抽象为具备感知、记忆、决策、动作和人工协同接口的"
    "轻量级边缘 Agent，并给出对应的模块化封装方案、运维机制和论文撰写结构。"
)

KEYWORDS_CN = "猪只计数；边缘智能；Agent 封装；自治运维；人机协同"

ABSTRACT_EN = (
    "This paper studies how to encapsulate an already deployed pig passage counting project "
    "into an edge agent with autonomous operation and human-in-the-loop review capabilities. "
    "The original visual core is retained, including YOLOv8-based pig detection, ByteTrack-"
    "based multi-object tracking, and a three-line bidirectional counting strategy. On top of "
    "this visual core, an agent-oriented layer is introduced with state memory, policy "
    "decision, action execution, anomaly auditing, and human review interfaces. Therefore, the "
    "system is transformed from a set of counting scripts into a lightweight edge agent that "
    "is able to operate continuously, expose maintainable interfaces, and support paper-"
    "oriented evaluation. Based on the existing experiments from dataset groups four, five, "
    "and six, 18 labeled videos in group four achieve an exact-match rate of 88.89% and a "
    "mean absolute error of 0.333 under the filename-label protocol. After correcting the "
    "ground-truth value of 17-37tou.mp4 to 31 through human review, the exact-match rate "
    "remains 88.89% while the mean absolute error decreases to 0.111, and the cumulative "
    "predicted count matches the corrected cumulative ground truth at 648. For long-video and "
    "high-count scenarios, group five yields total_line 2187 and valid_traj 2193, while the "
    "three videos in group six produce 2184/2188, 568/565, and 979/975 respectively. The paper further organizes the project into an "
    "agent-oriented framework with perception, memory, decision, action, and human review, "
    "thereby providing a stronger academic narrative for system presentation and future work."
)

KEYWORDS_EN = "pig counting; edge intelligence; agent encapsulation; autonomous operations; human in the loop"


TABLES = {
    "table1": {
        "title": "表1 由视觉项目到边缘 Agent 的模块映射",
        "headers": ["Agent 组成", "当前项目中的对应实现", "建议新增封装", "论文价值"],
        "rows": [
            ["Perception", "YOLOv8 检测、蓝色目标过滤、ByteTrack 跟踪", "统一感知接口，输出检测框、轨迹、计数事件", "保留算法核心，明确 Agent 输入输出"],
            ["Memory", "历史计数、轨迹报告、CSV/TXT 输出", "增加状态缓存、异常事件表、特殊样本注册表", "支撑长期运行和可追溯分析"],
            ["Decision", "现有脚本以内嵌规则为主", "独立策略模块，判断冻结、失速、计数异常和人工介入条件", "体现 Agent 自主性"],
            ["Action", "Reset、导出 CSV、部署脚本、网页监控", "增加自动重连、日志归档、告警与任务派发动作", "形成完整闭环"],
            ["Human Loop", "当前主要依靠口头说明和人工理解", "建立复核登记表、证据链接、论文口径字段", "解决特殊样本争议"],
        ],
    },
    "table2": {
        "title": "表2 自治运维场景下的异常类型与响应动作",
        "headers": ["异常类别", "触发信号", "建议阈值/规则", "Agent 动作", "人工协同方式"],
        "rows": [
            ["视频流冻结", "连续多周期 frame_count 不变", "例如 3 到 5 秒无新帧", "重连数据源、记录事件、标记恢复次数", "必要时人工确认摄像头状态"],
            ["推理性能下降", "fps_inference 明显低于历史基线", "跌幅超过 30%", "触发性能告警、切换低负载参数", "运维人员复核硬件负载"],
            ["计数异常突变", "单位时间 total_count 增量异常", "超过经验阈值", "保留上下文片段并进入人工复核队列", "人工判断是否为聚集或遮挡场景"],
            ["轨迹失稳", "valid_traj 与 total_line 偏差持续扩大", "偏差持续存在若干窗口", "记录风险样本、导出证据 CSV", "人工确认是否属于特殊样本"],
            ["结果争议", "标签值与计数值不一致但业务上可解释", "人工登记说明", "将样本写入 review registry", "形成论文口径统计"],
        ],
    },
    "table3": {
        "title": "表3 数据集与实验样本组成",
        "headers": ["数据组", "样本数量", "是否纳入论文统计", "用途", "备注"],
        "rows": [
            ["第四组", "18 个视频", "是", "主准确性评估", "仅保留有真实值样本"],
            ["第五组", "1 个长视频", "否", "长时连续运行验证", "无统一真值，仅作运行分析"],
            ["第六组", "3 个视频", "否", "高计数与长视频验证", "无统一真值，仅作运行分析"],
        ],
    },
    "table4": {
        "title": "表4 第四组带标注样本的量化结果",
        "headers": ["统计口径", "样本数/总量", "Exact-match", "MAE", "备注"],
        "rows": [
            ["文件名标签口径", "18", "16/18 (88.89%)", "0.333", "17-37头.mp4 按命名值 37 统计"],
            ["人工复核真值口径", "18", "16/18 (88.89%)", "0.111", "17-37头.mp4 真值修正为 31，预测值为 32"],
            ["修正后累计总数", "648 vs 648", "-", "-", "预测总数与修正后真实值总数一致"],
        ],
    },
    "table5": {
        "title": "表5 长视频与高计数场景结果",
        "headers": ["数据组", "视频", "total_line", "valid_traj", "耗时 / s", "说明"],
        "rows": [
            ["第五组", "long_video.mp4", "2187", "2193", "1056.6", "长时连续运行样本"],
            ["第六组", "1.mp4", "2184", "2188", "600.9", "高密度计数样本"],
            ["第六组", "2.mp4", "568", "565", "157.0", "中等时长样本"],
            ["第六组", "3.mp4", "979", "975", "245.3", "拼接长视频样本"],
        ],
    },
    "table6": {
        "title": "表6 面向 Agent 落地的分阶段实施路线",
        "headers": ["阶段", "主要工作", "对应输出", "论文与工程意义"],
        "rows": [
            ["第一阶段", "将检测、跟踪、计数主链统一封装为 perception service", "统一输入输出接口、标准事件格式", "形成 Agent 内核"],
            ["第二阶段", "补齐状态记忆、异常日志、人工复核登记表", "health log、review registry、evidence package", "建立自治运维与人机协同能力"],
            ["第三阶段", "引入策略引擎和动作执行层，联动网页监控与部署脚本", "自动重连、告警、归档、评测报表", "把项目升级为可答辩、可扩展的系统研究"],
        ],
    },
}


@dataclass
class FigureSpec:
    figure_id: str
    caption: str
    filename: str
    width_cm: float


FIGURES = {
    "fig1": FigureSpec("fig1", "图1 猪只计数系统的 Agent 化封装总体架构", "fig1_agent_architecture.png", 15.5),
    "fig2": FigureSpec("fig2", "图2 边缘 Agent 的感知-记忆-决策-动作-人工协同闭环", "fig2_agent_loop.png", 15.5),
    "fig3": FigureSpec("fig3", "图3 三线双向计数机制示意图", "fig3_counting_strategy.png", 15.0),
    "fig4": FigureSpec("fig4", "图4 第四组带标注样本的结果对比与复核口径分析", "fig4_results_analysis.png", 15.5),
}


SECTIONS = [
    {
        "heading": "1 引言",
        "blocks": [
            {"type": "paragraph", "text": "猪只转群、分栏和通道放行场景中的数量统计直接关系到养殖过程管理、异常追溯与生产调度。传统人工计数方式需要人工长时间盯守通道视频或现场目测，不仅劳动成本高，而且容易受到遮挡、折返、拥挤、光照变化和长时间疲劳等因素影响。对于课程设计或实验项目而言，仅仅做出一个能够输出数字的计数程序往往已经足够；但若希望将项目提升为论文成果，就必须进一步回答“系统如何长期稳定运行”“异常情况下如何处理”“有争议样本如何进入学术评价口径”等问题。"},
            {"type": "paragraph", "text": "当前项目已经具备较完整的视觉主链，包括基于 YOLOv8 的猪只检测、基于 ByteTrack 的多目标跟踪、基于三条竖线的双向计数规则，以及面向 Atlas 200I DK A2 板端的部署与网页监控功能。这说明项目本身并不是一个简单的单脚本 demo，而是一个拥有离线评测、边缘部署、实时展示和历史结果输出能力的系统化工程。问题在于：现有代码结构仍然以“若干脚本协作”为主，并未在论文层面被抽象成一个具有感知、记忆、决策、动作和人工协同接口的 Agent。"},
            {"type": "paragraph", "text": "因此，本文的核心目标不是重新发明一个完全不同的算法，而是对现有猪只计数系统进行 Agent 化重构与论文化表达。本文所说的 Agent 并不等于大语言模型，也不意味着让语言模型逐帧参与计数决策；相反，本文强调的是一种面向边缘计算和工程落地的轻量级自治代理思想，即在保留已有视觉内核的前提下，将状态监控、异常判定、动作执行、日志审计和人工复核能力以模块方式封装在系统之外，使其能够持续运行、解释自身行为并支持人机协同。"},
            {"type": "paragraph", "text": "基于这一思路，本文的主要贡献可以概括为四点：第一，将已有 YOLOv8 + ByteTrack + 三线计数系统梳理为一个可论文化描述的边缘视觉主链；第二，将视觉主链进一步抽象为具备感知、记忆、决策、动作和人机协同接口的 Agent 框架；第三，提出自治运维与人工复核两类外围机制，以解决长期运行稳定性和特殊样本口径不一致问题；第四，基于现有第四、五、六组实验结果，对系统的准确性、长时运行能力和 Agent 化价值进行定量与定性分析。"},
        ],
    },
    {
        "heading": "2 相关工作",
        "blocks": [
            {"type": "paragraph", "text": "在目标检测方向，YOLO 系列模型因其单阶段结构、较高推理效率和较强工程易用性，被广泛应用于边缘视觉部署场景。现有项目采用 Ultralytics 提供的 YOLOv8 作为猪只单类别检测器，其优点在于训练、推理和导出流程统一，适合与已有训练权重和工程代码快速集成[1]。对于本文而言，YOLOv8 并不是全部创新来源，但它构成了 Agent 感知层的关键基础。"},
            {"type": "paragraph", "text": "在多目标跟踪方向，SORT、DeepSORT 和 ByteTrack 构成了经典的技术演化链。SORT 使用卡尔曼滤波和匈牙利匹配完成轻量级实时跟踪[4]；DeepSORT 在此基础上引入外观特征，提高了复杂场景下的身份保持能力[5]；ByteTrack 则通过关联低分检测框改善遮挡与漏检条件下的轨迹连续性[2]。对于猪只通道计数场景而言，跟踪的价值不在于精确识别每一头猪的身份，而在于保证跨帧轨迹稳定，从而让穿线计数规则具有可重复性。"},
            {"type": "paragraph", "text": "在养殖场景应用方面，已有研究已经证明基于目标检测与跟踪的视频计数方法具有现实价值。例如，EmbeddedPigCount 讨论了在嵌入式板卡上完成猪只视频计数的可能性[3]。此外，一些畜牧视觉研究还关注行为识别、姿态估计、体尺测量和个体监测等问题，说明边缘视觉系统在现代养殖中具有持续扩展空间。与这些工作相比，本文并不追求提出一套全新的检测模型，而是强调如何将既有视觉内核封装为更符合论文表达和工程运维需求的边缘 Agent。"},
            {"type": "paragraph", "text": "值得指出的是，许多课程项目或工程项目的论文写作常常停留在“使用某某算法完成某某任务”的层面，但对系统长期运行和异常处理的讨论较少。事实上，真实部署中的大量价值恰恰来自这些外围机制。因此，本文在相关工作基础上进一步强调 Agent 封装的必要性：一方面，视觉算法提供感知能力；另一方面，围绕视觉算法建立起来的状态记忆、策略判断、动作执行和人工协同流程，决定了系统是否真正具备自治运行属性。"},
        ],
    },
    {
        "heading": "3 问题定义与 Agent 化目标",
        "blocks": [
            {"type": "paragraph", "text": "本文面向的具体问题定义为：给定一个固定视角的视频流或 RTSP 摄像头输入，系统需要在猪只从右侧入口区域进入、经过等待区并最终穿过通道左侧出口区域的过程中，对通过数量进行实时统计，同时输出可追溯的轨迹报告和运行状态信息。由于猪只可能出现折返、停留、拥挤和相互遮挡，单纯依赖逐帧检测数量并不能得到稳定结果，因此需要结合跨帧跟踪与时序计数规则。"},
            {"type": "paragraph", "text": "在此基础上，本文进一步给出 Agent 化目标：系统不仅需要回答“当前有多少头猪通过”，还需要回答“这个结果是否可信”“当前是否发生了异常”“系统采取了什么动作”“哪些样本需要人工确认”。也就是说，Agent 的职责应包括四个层面：第一，围绕视觉主链持续感知场景状态；第二，维护对计数结果、历史趋势和异常信号的内部记忆；第三，根据既定规则自动做出重连、重置、记录或上报决策；第四，在结果存在争议时将样本交给人工复核，并将人工意见重新纳入系统。"},
            {"type": "paragraph", "text": "从形式化角度看，可将 Agent 抽象为五元组 A = (O, M, P, U, H)，其中 O 表示观测集合，包括检测框、轨迹、FPS、frame_count、total_count、异常事件等；M 表示状态记忆，包括历史计数、轨迹摘要、风险样本队列和人工复核登记；P 表示策略规则，用于判断是否触发异常、是否应自动重连以及是否需要人工介入；U 表示动作集合，如 reset、reconnect、export、alert；H 表示 human-in-the-loop 接口，用于修正特殊样本并形成论文统计口径。"},
            {"type": "figure", "figure_id": "fig1"},
            {"type": "table", "table_id": "table1"},
        ],
    },
    {
        "heading": "4 Agent 封装架构设计",
        "blocks": [
            {"type": "paragraph", "text": "基于上述问题定义，本文将系统划分为视觉主链层、状态记忆层、策略决策层、动作执行层和人工复核层五个部分。视觉主链层负责完成视频流获取、目标检测、蓝色干扰过滤、ByteTrack 关联和三线计数，是整个系统的感知与执行内核。状态记忆层负责维护 frame_count、fps_inference、三线计数值、有效轨迹数、历史趋势、恢复次数和争议样本记录，从而为后续决策提供上下文。"},
            {"type": "paragraph", "text": "策略决策层是 Agent 化封装的关键所在。与传统脚本将判断逻辑散落在主函数内部不同，本文建议将健康状态评价、异常阈值判定和人工复核规则从视觉内核中剥离出来，形成独立的 policy engine。该层并不直接改变检测器和跟踪器的计算方式，而是通过读取状态记忆层的指标，对系统运行状态进行更高层的判断。例如，当一段时间内 frame_count 无增长时，可判定视频流冻结；当 fps_inference 显著下降时，可触发性能预警；当 total_line 与 valid_traj 的差异在滑动窗口内持续扩大时，可标记该样本为高风险样本。"},
            {"type": "paragraph", "text": "动作执行层负责把决策转换为可执行操作。现有项目实际上已经具有部分动作接口，例如网页侧提供 Reset 功能、批处理脚本可以自动复跑视频、部署脚本可以向 Atlas 板端上传文件和数据。将这些能力统一纳入 Agent 后，可以把动作标准化为若干原语，包括 reconnect_stream、reset_counter、dump_csv、save_evidence、raise_alert 和 enqueue_review 等。这样一来，论文中的 Agent 不再只是抽象概念，而是具有可列举输入、状态和动作集合的完整系统。"},
            {"type": "paragraph", "text": "人工复核层则解决了工程项目在学术表达中经常遇到的一个痛点：少量特殊样本的存在会严重影响评价结果，但这些样本往往并不是简单的算法错误，而是受现场环境、标签定义、猪只折返和业务统计口径影响。本文认为，与其在论文中回避这些样本，不如显式引入 review registry，对每个争议样本记录原始误差、人工说明、证据链接和论文口径字段。这样既能保留原始统计的客观性，又能展示系统具备现实部署中的人机协同能力。"},
            {"type": "figure", "figure_id": "fig2"},
            {"type": "table", "table_id": "table2"},
        ],
    },
]


SECTIONS.extend([
    {
        "heading": "5 关键算法与策略机制",
        "blocks": [
            {"type": "paragraph", "text": "5.1 检测与过滤。现有系统使用 YOLOv8 完成猪只单类别检测，并将输出检测框转换为 [x1, y1, x2, y2, score] 的统一格式。考虑到现场存在蓝色桶、管道等干扰物，系统在检测之后进一步执行蓝色目标过滤。过滤函数根据检测框区域的 BGR 平均值判断是否为明显蓝色目标，从而在不引入额外模型的情况下抑制一部分误检。这一规则虽然简单，但对通道类场景具有较强工程针对性，也适合作为 Agent 感知层中的场景先验。"},
            {"type": "paragraph", "text": "5.2 跟踪与轨迹有效性判定。系统使用 ByteTrack 维护跨帧目标 ID，对每个激活轨迹记录历史区域、位置序列、置信度、丢失帧和恢复次数等信息。与只输出最终总数相比，这些轨迹级别的中间状态是 Agent 化封装的重要资源，因为它们既可以用于解释当前结果，也可以用于支持异常检测和人工复核。本文认为，当一个系统能够回答“为什么这一头猪被计入/未计入”时，它才具备面向论文和答辩的可解释性。"},
            {"type": "paragraph", "text": "5.3 三线双向计数。现有实现按照画面宽度设置三条竖线：Line0 位于外部区域中间，Line1 位于 OUT 与 WAIT 的边界，Line2 位于 WAIT 与 ENTRY 的边界。对于每个 track_id，若轨迹中心点从右向左越过某条线，则对应计数加 1；若从左向右越过，则对应计数减 1。最终总数按照 C_total = round((C0 + C1 + C2) / 3) 计算。之所以保留三条线而不是单线，是因为多线汇总能够在一定程度上抑制单线误触发，尤其适合猪只折返和局部遮挡较多的通道场景。"},
            {"type": "figure", "figure_id": "fig3"},
            {"type": "paragraph", "text": "5.4 Agent 健康评估机制。为了让系统具备自治运行能力，本文建议引入一个轻量级健康评分 H。设 s_stream 表示视频流是否持续更新，s_fps 表示推理性能相对基线的保持程度，s_consistency 表示 total_line 与 valid_traj 的一致性，则可用 H = w1*s_stream + w2*s_fps + w3*s_consistency 对当前运行状态进行打分。当 H 连续低于阈值时，Agent 触发恢复动作或将样本送入人工复核队列。这一机制不要求复杂学习模型，依赖规则即可实现。其论文价值在于把原本零散的工程判断上升为可描述、可调参、可评价的自治策略。"},
            {"type": "paragraph", "text": "5.5 人机协同复核机制。对于带标签视频，定义原始误差 E_raw = C_pred - C_gt，用于反映系统输出与命名标签之间的直接偏差；当人工复核发现文件命名值本身不能代表真正的统计目标时，则进一步给出修正真值 C_gt*，并据此计算复核误差 E_review = C_pred - C_gt*。这种做法并不是简单把误差清零，而是把人工知识用于修正真值定义本身。通过同时保留 E_raw、E_review、复核说明与证据来源，系统既能保留原始统计，也能避免错误标签持续污染论文结果。"},
        ],
    },
    {
        "heading": "6 实验设置",
        "blocks": [
            {"type": "paragraph", "text": "本文实验基于现有项目仓库中的历史结果展开，未对视觉主链作破坏性修改。PC 端批处理脚本调用 track_and_count.py 对第四、五、六组视频进行离线复跑；板端版本则面向 Atlas 200I DK A2 提供 NPU 推理和网页监控入口。根据项目文档，PC 端采用训练得到的 best.pt 权重执行 YOLOv8 推理，板端使用量化后的 .om 模型。"},
            {"type": "paragraph", "text": "在论文统计口径上，本文仅采用第四组中 18 个具有明确真实值的视频作为主评估集，其余未给出真实值的第四组视频一律不纳入论文数据统计，可视为评测外样本。第五组包含 1 个长视频，第六组包含 3 个高计数或长时样本，主要用于分析持续运行能力而非准确率。这种口径收紧虽然减少了样本数，但能够保证论文中的每一个准确性指标都对应可核验的真实值。"},
            {"type": "paragraph", "text": "在评价指标方面，本文同时报告 exact-match 和平均绝对误差 MAE。对于第四组 18 个有真实值样本，exact-match 表示预测总数与标签完全相等的比例，MAE 表示预测值与标签之差的绝对值平均。对于第五、六组以及第四组中缺乏真实值的视频，本文不计算准确率，而是仅报告 total_line、valid_traj 和耗时，以分析系统在长时连续运行中的稳定性和统计一致性。"},
            {"type": "paragraph", "text": "除了传统准确性指标，本文还特别强调 Agent 相关评价维度。对于一个面向长期运行的边缘代理而言，仅有准确率并不足以说明系统成熟度，还需要关注运行中断次数、重启与恢复开销、异常事件记录完整性、导出证据的可用性以及人工复核流程是否闭环。因此，本文虽然当前尚未拿到完整的自治运维日志，但在论文写作结构上已经为这部分指标预留了清晰位置。"},
            {"type": "paragraph", "text": "在参数设置上，离线脚本默认采用 conf_thres = 0.5、track_thresh = 0.5、out_ratio = 0.45 和 wait_ratio = 0.25。这样的设置与现有项目代码保持一致，有利于保证论文叙述与实际运行结果相符。对于 Agent 层来说，这些参数不仅是视觉主链的超参数，也是策略层可以观测和调整的状态对象。后续若继续完善系统，可进一步讨论参数漂移监测与参数自适应问题。"},
            {"type": "paragraph", "text": "在输出形式上，系统不仅生成最终总数，还会输出带标注视频、ID 事件 CSV、轨迹报告 CSV、状态变化 TXT 和汇总 CSV。这些输出对论文而言意义重大，因为它们构成了 Agent 记忆层和审计层的基础材料。与只能给出单个数字的黑盒式系统相比，具备多粒度中间输出的系统更适合开展误差分析、特殊样本追踪和人机协同复核。"},
            {"type": "table", "table_id": "table3"},
        ],
    },
    {
        "heading": "7 实验结果与分析",
        "blocks": [
            {"type": "paragraph", "text": "7.1 第四组带标注样本结果。根据已有 batch_rerun_group4_results.csv，18 个带标签视频中有 16 个样本达到 exact-match，文件名标签口径下 exact-match 为 88.89%，MAE 为 0.333。从结果明细看，主要偏差集中在 15-47头.mp4 和 17-37头.mp4 两个样本。前者维持 -1 的误差不变；后者原本按文件命名被记为 37，但人工复核表明其中有 6 个目标在视频开始时从左侧反向进入，不应计入正常方向通行目标，因此修正后的真实值应为 31。"},
            {"type": "paragraph", "text": "7.2 人工复核真值口径分析。将 17-37头.mp4 的真实值修正为 31 后，系统在该视频上的预测值为 32，因此复核后误差为 +1，而不是原来的 -5。由此，第四组 18 个样本在人工复核真值口径下的 exact-match 仍为 88.89%，但 MAE 从 0.333 降低到 0.111。进一步统计可知，18 个样本的累计预测总数为 648，修正后的累计真实值总数同样为 648，总量误差为 0。这个结果说明，人工复核在本研究中的作用不是“把错误改成正确”，而是纠正错误标签，使评价结果更接近实际业务场景。"},
            {"type": "paragraph", "text": "从样本层面来看，17-37头.mp4 的价值不只是让一个数值从 -5 变成 +1，而在于它揭示了一个非常典型的论文写作问题：真实业务场景中的计数目标并不总能被单一文件名标签穷尽。当猪只在入口区和等待区发生复杂交错、折返或反向通行时，命名值与真正应统计的目标集合可能并不一致。将这类样本显式写入 review registry，并在论文中说明“如何修正真值”，要比简单说它是特殊样本更严谨。"},
            {"type": "figure", "figure_id": "fig4"},
            {"type": "table", "table_id": "table4"},
            {"type": "paragraph", "text": "7.3 长视频与高计数场景分析。第五组 long_video.mp4 的 total_line 为 2187，valid_traj 为 2193，处理耗时 1056.6 秒；第六组三个视频分别得到 2184/2188、568/565 和 979/975 的两类统计值。可以看到，在这些高计数和长视频场景中，line-based 统计与轨迹有效性统计的差异保持在较小范围内，表明视觉主链具备一定的连续运行能力。这也为后续在 Agent 层增加异常监测与自动恢复提供了基础，因为只有当感知内核本身相对稳定时，外围自治机制才有意义。"},
            {"type": "paragraph", "text": "长视频结果对论文的另一个启发在于：系统研究不一定要把所有实验都写成传统意义上的分类准确率或检测 mAP。对于边缘 Agent 而言，连续 600 秒到 1000 秒级别的稳定处理、本地板端长时间运行和计数曲线的整体平滑性，同样构成重要实验现象。换句话说，论文不应只问“有没有算对”，还应问“能不能持续地算”“系统在长时间运行中是否具备自我感知和恢复的潜力”。"},
            {"type": "table", "table_id": "table5"},
            {"type": "paragraph", "text": "7.4 误差来源分析。结合已有输出文件，可以把误差来源粗略分为四类：第一类是检测层面误检与漏检；第二类是跟踪层面 ID 切换、短时丢失与恢复；第三类是计数层面单线误触发或折返路径带来的汇总波动；第四类是标签与业务统计口径不一致导致的“伪误差”。前两类偏向视觉内核问题，第三类偏向规则设计问题，第四类则只有通过人机协同复核才能合理处理。这样的分层误差分析，正是 Agent 化论文比传统项目报告更有研究味道的地方。"},
            {"type": "paragraph", "text": "7.5 结果讨论。总体来看，现有项目在工程实现上已经明显超出传统课程作业范畴，具备完整的数据流、运行入口和结果输出形式。它的主要不足不在于“没有算法”，而在于“还没有被抽象成一个更强的学术叙事”。Agent 化封装恰好提供了这样一种叙事方式：视觉主链负责感知与计数，自治运维层负责长期运行稳定性，人机协同层负责争议样本处理。三者结合后，论文不再只是算法流程图，而是一个更像系统研究的整体方案。"},
        ],
    },
    {
        "heading": "8 Agent 化价值讨论",
        "blocks": [
            {"type": "paragraph", "text": "从论文写作角度看，Agent 化最大的价值在于把原本分散在多个脚本和说明文档中的工程能力，重新组织为统一的研究对象。对于答辩老师或评审而言，一个“边缘 Agent”比一个“若干脚本组成的项目”更容易被理解和比较，因为前者能够清晰说明自己的感知输入、内部记忆、决策规则、执行动作和人机接口。"},
            {"type": "paragraph", "text": "从工程落地角度看，Agent 化并不要求立即引入复杂的大模型或强化学习机制。对当前项目而言，最现实的做法是保持现有 YOLOv8 + ByteTrack 视觉核心不变，将其视作稳定的感知执行单元；再在外部增加轻量的健康检测、异常记录、自动恢复和人工复核模块。这样的改造成本低、与现有代码兼容性高，同时又足以支撑论文中“自治运维”和“人机协同”的主题。"},
            {"type": "paragraph", "text": "从学术表达角度看，Agent 化还能带来更多可分析指标。例如，可以单独统计系统的异常发现率、自动恢复成功率、人工介入次数、复核通过率和审计覆盖率；这些指标虽然不属于传统检测或跟踪准确率，但在真实部署中往往更能体现系统成熟度。对于后续继续扩展论文而言，也可以在此基础上进一步加入参数自适应、主动学习或异常片段自动摘要等更高级的能力。"},
            {"type": "paragraph", "text": "从系统答辩角度看，Agent 化还有一个实际优势：它能帮助作者把“代码做了什么”和“系统为什么这样设计”区分开来。很多工程项目在答辩中会陷入逐行解释脚本的困境，而 Agent 架构允许作者站在更高层抽象上回答问题，例如为什么需要记忆层、为什么要保留人工复核接口、为什么异常日志是系统的一部分。这样的回答方式更接近正式论文中的系统设计论证。"},
            {"type": "paragraph", "text": "从课程向论文迁移的角度看，Agent 化也有助于整理研究边界。视觉主链负责“看见与数出”，策略层负责“判断是否可信”，动作层负责“执行与恢复”，人机层负责“争议归因与知识回写”。当各层边界清晰后，后续无论是继续写毕业设计、开题报告还是投稿，都可以围绕某一层展开深化，而不必从零重构整个工程。"},
            {"type": "paragraph", "text": "进一步地，Agent 封装还为项目开源与复现带来好处。若未来希望对外展示该系统，只暴露若干脚本入口往往不利于他人理解；而若以 Agent 形式组织文档和接口，则可以把输入、状态、动作和输出明确列出，降低学习成本。对学术社区而言，一个更容易复现和理解的系统，也更容易获得认可。"},
            {"type": "paragraph", "text": "当然，本文的 Agent 化封装仍然存在局限。首先，当前自治策略主要依赖规则和阈值，尚未形成基于学习的决策机制；其次，人工复核部分目前仍以登记表和说明为主，缺少更加标准化的工具链；再次，现有数据集规模较小，且公开性不足，不利于做更强的横向比较。但即便如此，本文已经为后续从工程项目走向正式学术投稿奠定了较完整的框架基础。"},
        ],
    },
    {
        "heading": "9 从现有项目到 Agent 系统的实施路线",
        "blocks": [
            {"type": "paragraph", "text": "若要把当前项目真正落到“Agent 系统”的实现层面，而不仅仅停留在论文叙事上，最合理的策略不是推倒重来，而是沿着现有代码结构逐步重构。第一步应围绕视觉主链做接口统一，把离线脚本、实时监控脚本和 NPU 版本中的共性输出抽象成统一事件格式，例如 detection event、track event、count event 和 health event。这样可以最大限度复用现有视觉代码，同时为上层策略引擎提供稳定输入。"},
            {"type": "paragraph", "text": "第二步应建立状态记忆与人工复核的数据载体。具体而言，可将现有 summary.csv、trajectory_report.csv、state_changes.txt 等输出进一步整理为 machine-readable 的状态对象，并新增 review registry、异常事件日志和 evidence package 索引。这样做的结果是，Agent 不再只在运行时产生瞬时判断，而是具有可追溯的长期记忆。论文中关于人机协同和自治运维的讨论也会因此变得更扎实。"},
            {"type": "paragraph", "text": "第三步应将网页监控和部署脚本纳入动作层。当前项目已经存在 reset、下载 CSV、部署到板端等动作入口，但这些能力尚未被统一视为 Agent action。若后续将其标准化，并为每个动作记录触发原因、执行时间和结果状态，就能形成真正的感知-决策-动作闭环。对于论文而言，这意味着系统不再只是“看见并输出”，而是“发现问题并采取动作”。"},
            {"type": "paragraph", "text": "第四步应完善评测协议，将传统准确性评估与 Agent 化评测并行呈现。前者继续使用 exact-match 和 MAE；后者则加入 anomaly recall、auto-recovery success rate、review acceptance rate 等指标。即使这些指标在第一版论文中尚未全部给出，也应在设计上明确其定义与采集方式。这样可以为后续升级版论文留出自然扩展空间。"},
            {"type": "paragraph", "text": "总的来说，项目的 Agent 化不是另起炉灶，而是在现有工程基础上做结构化封装与指标补全。这样的路线既符合课程项目向研究型系统演进的实际节奏，也能让论文从一开始就具备后续延展能力。"},
            {"type": "table", "table_id": "table6"},
        ],
    },
    {
        "heading": "10 可扩展实验设计与研究问题",
        "blocks": [
            {"type": "paragraph", "text": "若后续继续扩展本文工作，最值得优先补充的是消融实验。现有系统可以自然形成若干对比版本，例如：仅使用检测结果而不使用 ByteTrack 的版本、只使用单条计数线的版本、去除蓝色过滤规则的版本、去除人工复核口径的版本，以及去除自治运维策略的版本。通过这些对比，可以更明确地说明视觉内核中的每一个模块和 Agent 外围机制分别带来了什么收益。"},
            {"type": "paragraph", "text": "第二类值得补充的实验是自治运维有效性实验。可以人为构造视频流中断、帧冻结、推理性能下降和计数突变等异常场景，统计 Agent 对异常的发现率、平均响应时间、自动恢复成功率和误告警率。与传统准确率实验相比，这类实验更能体现“封装成 Agent”之后系统能力的变化，也更符合论文题目中的自治运维主题。"},
            {"type": "paragraph", "text": "第三类实验是人机协同效率实验。除了统计复核后准确率的变化，还可以记录人工复核一个争议样本所需的时间、人工复核意见被系统采纳的比例、以及 review registry 对最终论文口径的影响程度。这样可以从更完整的角度说明，人机协同并不是简单的“手动改结果”，而是把专家知识以可追溯形式纳入系统闭环。"},
            {"type": "paragraph", "text": "第四类实验是跨场景泛化实验。当前数据主要来自同一类通道场景，后续可以增加不同光照条件、不同摄像机安装高度、不同栏舍通道宽度和不同猪只密度条件下的视频，从而验证 Agent 架构是否仍然适用。即使视觉主链的准确率在不同场景中有所波动，只要自治运维与人机协同接口设计得当，系统依然可能保持较高的总体可用性。"},
            {"type": "paragraph", "text": "第五类值得讨论的问题是研究有效性边界。本文目前更多是一篇系统整理与封装导向的论文，而不是纯算法创新论文，因此在投稿或答辩时应主动说明创新点位于系统抽象、Agent 封装、自治运维与人机协同机制，而非全新网络结构。只要把这一定位讲清楚，论文就能在合理预期下成立，避免被误读为算法创新不足。"},
            {"type": "paragraph", "text": "第六类值得补充的是复现性实验。由于当前项目含有 PC 端脚本、板端部署包、网页监控入口和多种输出文件格式，因此完全可以把复现性本身作为论文的一部分进行讨论，例如是否能够在另一台设备上复现实验流程、是否能依据文档复原目录结构、是否能从历史结果中重新生成统计表。对于系统类论文而言，复现性往往与可信度直接相关。"},
            {"type": "paragraph", "text": "第七类值得考虑的是部署成本与收益评估。虽然这不是传统视觉论文的核心指标，但对于边缘 Agent 系统而言，设备成本、单次部署耗时、维护频率和人工节省量都会影响系统的实际价值。若后续有条件，可以将“人工计数方案”与“Agent 方案”在时间投入、异常追溯效率和长期稳定性上做简单对比，从而让论文更接近真实应用场景。"},
            {"type": "paragraph", "text": "最后，还应注意数据与伦理边界。本文场景主要围绕猪只通道计数，不涉及个体隐私识别，但仍需要在论文中说明数据采集来源、使用目的和场景约束，避免给人留下“系统可以无边界泛化”的误解。对于课程或毕业设计论文而言，这样的边界意识虽然不属于算法创新，却能显著提升论文的完整性与规范性。"},
            {"type": "paragraph", "text": "还可以进一步加入论文写作层面的对照实验设计，例如分别以“纯视觉计数系统”和“Agent 化边缘系统”两种叙事方式重构同一项目，再比较二者在问题定义、指标体系、图表组织和答辩说服力上的差异。虽然这类比较不属于严格数值实验，但它能帮助作者明确：Agent 化并不是简单换个名字，而是对项目研究对象、系统边界和评价方式的整体升级。"},
            {"type": "paragraph", "text": "如果后续确实准备投稿，还应在这一节末尾明确论文适合的稿件类型，例如系统实现型、应用研究型或工程实践型。提前确定稿件定位，可以帮助后续有针对性地补实验，而不是盲目追求并不匹配当前项目阶段的算法创新指标。"},
            {"type": "paragraph", "text": "这一点对课程项目升级为论文尤其关键，也最容易被忽视。"},
        ],
    },
    {
        "heading": "11 结论",
        "blocks": [
            {"type": "paragraph", "text": "本文围绕现有猪只智能计数项目，提出了一种面向 Agent 封装的边缘视觉系统研究路径。核心思路在于：保留 YOLOv8 检测、ByteTrack 跟踪和三线双向计数构成的视觉主链，将其视作 Agent 的感知执行内核；再引入状态记忆、策略决策、动作执行和人工复核机制，使系统具备自治运维和人机协同能力。实验结果表明，现有系统在第四组带标注样本上已具备较好的准确性基础；在第五组 long_video.mp4 上得到 total_line 2187、valid_traj 2193；在第六组三个视频上分别得到 2184/2188、568/565 和 979/975 的结果，说明系统在长视频与高计数场景中也具备持续运行能力。"},
            {"type": "paragraph", "text": "后续工作可从三个方向继续推进：第一，将自治运维层真正写入代码，包括异常日志、自动恢复和 review registry 等模块；第二，补充更多跨场景数据，形成更规范的数据集和更丰富的对比实验；第三，在 Agent 框架上继续叠加主动学习、参数自适应和更强的人机交互能力。通过这些工作，现有项目有望从课程级工程成果进一步发展为更完整的研究型系统。"},
        ],
    },
])


REFERENCES = [
    "[1] Ultralytics Team. Introducing the Ultralytics YOLOv8 model. Ultralytics Blog, 2023.",
    "[2] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV, 2022.",
    "[3] Kim J, Suh Y, Lee J, et al. EmbeddedPigCount: Pig Counting with Video Object Detection and Tracking on an Embedded Board. Sensors, 2022, 22(7): 2689.",
    "[4] Bewley A, Ge Z, Ott L, et al. Simple Online and Realtime Tracking. ICIP, 2016.",
    "[5] Wojke N, Bewley A, Paulus D. Simple Online and Realtime Tracking with a Deep Association Metric. ICIP, 2017.",
    "[6] Nasirahmadi A, Hensel O, Edwards S, Sturm B. Implementation of machine vision for detecting behaviour of cattle and pigs. Livestock Science, 2017.",
    "[7] Ma J, Li Y, Li W, et al. Computer Vision-Based Measurement Techniques for Livestock Body Dimensions and Weight: A Review. Agriculture, 2024.",
]


def load_csv_rows(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_group4_labeled():
    rows = load_csv_rows(GROUP4_CSV)
    labeled = [r for r in rows if r["actual"]]
    for row in labeled:
        row["actual_i"] = int(row["actual"])
        row["pred_i"] = int(row["total_line"])
        row["error_i"] = int(row["error_line"])
        row["corrected_actual_i"] = 31 if row["video"] == "17-37头.mp4" else row["actual_i"]
        row["corrected_error_i"] = row["pred_i"] - row["corrected_actual_i"]
    return labeled


def metrics_from_group4(rows):
    n = len(rows)
    raw_exact = sum(1 for r in rows if r["error_i"] == 0)
    raw_mae = sum(abs(r["error_i"]) for r in rows) / n
    corrected_exact = sum(1 for r in rows if r["corrected_error_i"] == 0)
    corrected_mae = sum(abs(r["corrected_error_i"]) for r in rows) / n
    pred_total = sum(r["pred_i"] for r in rows)
    corrected_total = sum(r["corrected_actual_i"] for r in rows)
    return {
        "raw_exact": raw_exact,
        "raw_exact_rate": 100.0 * raw_exact / n,
        "raw_mae": raw_mae,
        "corrected_exact": corrected_exact,
        "corrected_exact_rate": 100.0 * corrected_exact / n,
        "corrected_mae": corrected_mae,
        "pred_total": pred_total,
        "corrected_total": corrected_total,
    }


def generate_architecture_figure(path: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.18
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2.0, color="#334155"))

    box(0.4, 5.3, 2.0, 1.2, "Camera / Video", "#7dd3fc")
    box(2.9, 5.3, 2.2, 1.2, "Perception Core\nYOLOv8 + Filter", "#86efac")
    box(5.5, 5.3, 2.0, 1.2, "Tracking Core\nByteTrack", "#fde68a")
    box(8.0, 5.3, 2.0, 1.2, "Counting Core\nThree-line Rule", "#fdba74")
    box(10.5, 5.3, 2.5, 1.2, "Dashboard / Reports\nWeb UI + CSV", "#c4b5fd")
    box(2.7, 2.8, 2.4, 1.2, "State Memory\nCounts / IDs / FPS / Events", "#a7f3d0")
    box(5.7, 2.8, 2.4, 1.2, "Policy Engine\nHealth + Review Rules", "#f9a8d4")
    box(8.7, 2.8, 2.2, 1.2, "Action Layer\nReset / Reconnect /\nExport / Alert", "#fca5a5")
    box(11.3, 2.8, 2.2, 1.2, "Human Review\nSpecial-case Registry", "#bfdbfe")

    arrow(2.4, 5.9, 2.9, 5.9)
    arrow(5.1, 5.9, 5.5, 5.9)
    arrow(7.5, 5.9, 8.0, 5.9)
    arrow(10.0, 5.9, 10.5, 5.9)
    arrow(9.0, 5.3, 9.0, 4.1)
    arrow(4.0, 5.3, 4.0, 4.1)
    arrow(5.1, 3.4, 5.7, 3.4)
    arrow(8.1, 3.4, 8.7, 3.4)
    arrow(10.9, 3.4, 11.3, 3.4)
    arrow(12.0, 4.0, 11.8, 5.3)

    ax.text(6.9, 7.25, "Agent-oriented encapsulation over the existing visual pipeline", fontsize=15, ha="center", fontweight="bold")
    ax.text(6.9, 1.4, "The original counting project becomes an edge agent when perception, memory, decision, action,\n"
                      "and human review are explicitly exposed as modules.", fontsize=11, ha="center", color="#334155")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_agent_loop_figure(path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    nodes = [
        (5, 8.5, "Observe\nframes, tracks,\nline counts"),
        (8, 5.8, "Evaluate\nhealth, drift,\nanomaly"),
        (6.7, 2.4, "Act\nreconnect, reset,\nexport, alert"),
        (3.3, 2.4, "Audit\nlog events,\nstore evidence"),
        (2.0, 5.8, "Human Review\nspecial cases,\nlabel conflicts"),
    ]
    colors = ["#bfdbfe", "#fde68a", "#fca5a5", "#c4b5fd", "#a7f3d0"]

    for (x, y, text), color in zip(nodes, colors):
        circle = patches.Circle((x, y), 1.2, facecolor=color, edgecolor="#334155", linewidth=2, alpha=0.65)
        ax.add_patch(circle)
        ax.text(x, y, text, ha="center", va="center", fontsize=12, fontweight="bold")

    arrows = [
        ((5.9, 7.7), (7.2, 6.7)),
        ((7.8, 4.6), (7.0, 3.6)),
        ((5.6, 2.1), (4.3, 2.1)),
        ((2.7, 3.4), (2.2, 4.6)),
        ((2.9, 6.8), (4.1, 7.7)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2.2, color="#334155"))

    ax.text(5, 9.55, "Closed-loop edge agent for pig counting", ha="center", fontsize=16, fontweight="bold")
    ax.text(5, 0.7, "The agent does not replace the vision core. It wraps the core with stateful operation,\n"
                    "self-checking rules, and a human review interface.", ha="center", fontsize=11, color="#334155")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_counting_figure(path: Path):
    width, height = 1500, 700
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    colors = {
        "out": (255, 235, 205),
        "wait": (220, 252, 231),
        "entry": (219, 234, 254),
        "line0": (249, 115, 22),
        "line1": (14, 165, 233),
        "line2": (14, 165, 233),
        "text": (30, 41, 59),
    }

    x0 = 220
    x1 = 520
    x2 = 840

    draw.rectangle([120, 140, x1, 500], fill=colors["out"])
    draw.rectangle([x1, 140, x2, 500], fill=colors["wait"])
    draw.rectangle([x2, 140, 1380, 500], fill=colors["entry"])
    draw.text((190, 100), "Three-line bidirectional counting strategy", fill=colors["text"])
    draw.text((250, 170), "OUT", fill=colors["text"])
    draw.text((645, 170), "WAIT", fill=colors["text"])
    draw.text((1060, 170), "ENTRY", fill=colors["text"])

    for xpos, color, label in [(x0, colors["line0"], "Line0"), (x1, colors["line1"], "Line1"), (x2, colors["line2"], "Line2")]:
        draw.line([xpos, 120, xpos, 530], fill=color, width=8)
        draw.text((xpos - 30, 540), label, fill=color)

    track1 = [(1230, 280), (1000, 300), (760, 320), (460, 330), (250, 350)]
    for i in range(len(track1) - 1):
        draw.line([track1[i], track1[i + 1]], fill=(34, 197, 94), width=10)
    for p in track1:
        draw.ellipse([p[0] - 18, p[1] - 18, p[0] + 18, p[1] + 18], fill=(34, 197, 94))
    draw.text((1030, 240), "Right -> Left : +1", fill=(21, 128, 61))

    track2 = [(260, 420), (520, 410), (780, 430), (1060, 440)]
    for i in range(len(track2) - 1):
        draw.line([track2[i], track2[i + 1]], fill=(239, 68, 68), width=10)
    for p in track2:
        draw.ellipse([p[0] - 16, p[1] - 16, p[0] + 16, p[1] + 16], fill=(239, 68, 68))
    draw.text((260, 455), "Left -> Right : -1", fill=(185, 28, 28))

    draw.text((250, 610), "Count formula: C_total = round((C0 + C1 + C2) / 3)", fill=colors["text"])
    draw.text((250, 645), "Multi-line aggregation reduces accidental triggering on a single split line.", fill=colors["text"])
    img.save(path)


def generate_results_figure(path: Path):
    rows = load_group4_labeled()
    metrics = metrics_from_group4(rows)

    sample_idx = list(range(1, len(rows) + 1))
    actual = [r["actual_i"] for r in rows]
    corrected_actual = [r["corrected_actual_i"] for r in rows]
    predicted = [r["pred_i"] for r in rows]
    highlight_idx = next(i + 1 for i, r in enumerate(rows) if r["video"] == "17-37头.mp4")
    highlight_pred = next(r["pred_i"] for r in rows if r["video"] == "17-37头.mp4")
    highlight_actual = next(r["actual_i"] for r in rows if r["video"] == "17-37头.mp4")
    highlight_corrected = next(r["corrected_actual_i"] for r in rows if r["video"] == "17-37头.mp4")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.plot(sample_idx, actual, marker="o", linewidth=2.0, color="#2563eb", label="Filename label")
    ax.plot(sample_idx, corrected_actual, marker="^", linewidth=2.0, color="#dc2626", label="Reviewed ground truth")
    ax.plot(sample_idx, predicted, marker="s", linewidth=2.0, color="#16a34a", label="Predicted count")
    ax.scatter([highlight_idx], [highlight_actual], color="#2563eb", s=90, zorder=4)
    ax.scatter([highlight_idx], [highlight_corrected], color="#dc2626", s=90, marker="^", zorder=4)
    ax.scatter([highlight_idx], [highlight_pred], color="#dc2626", s=90, marker="s", zorder=4)
    ax.annotate("Label corrected\n37 -> 31", xy=(highlight_idx, highlight_corrected), xytext=(highlight_idx + 0.5, highlight_actual + 7),
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#dc2626", fontsize=10)
    ax.set_xlabel("Labeled sample index")
    ax.set_ylabel("Pig count")
    ax.set_title("Filename labels, reviewed ground truth, and predictions")
    ax.grid(alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    labels = ["Filename-label", "Reviewed-GT"]
    exact_rates = [metrics["raw_exact_rate"], metrics["corrected_exact_rate"]]
    maes = [metrics["raw_mae"], metrics["corrected_mae"]]
    x = range(len(labels))
    bars = ax2.bar(x, exact_rates, width=0.45, color=["#f59e0b", "#22c55e"], label="Exact-match (%)")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Exact-match (%)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_title("Metric change after reviewing the ground truth")
    for bar, val in zip(bars, exact_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.2f}", ha="center", fontsize=10)
    ax3 = ax2.twinx()
    ax3.plot(list(x), maes, color="#1d4ed8", marker="o", linewidth=2.2, label="MAE")
    ax3.set_ylabel("MAE")
    for xi, val in zip(x, maes):
        ax3.text(xi, val + 0.03, f"{val:.3f}", ha="center", color="#1d4ed8", fontsize=10)
    ax2.text(0.5, 12, f"Cumulative count: {metrics['pred_total']} vs {metrics['corrected_total']}", ha="center", fontsize=10, color="#334155")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_figures():
    generate_architecture_figure(FIG_DIR / FIGURES["fig1"].filename)
    generate_agent_loop_figure(FIG_DIR / FIGURES["fig2"].filename)
    generate_counting_figure(FIG_DIR / FIGURES["fig3"].filename)
    generate_results_figure(FIG_DIR / FIGURES["fig4"].filename)


def xml_text(text: str) -> str:
    return escape(text).replace("\n", "</w:t><w:br/><w:t>")


def paragraph_xml(text: str, *, align: str = "both", font: str = "宋体", latin: str = "Times New Roman",
                  size: int = 24, bold: bool = False, first_line: int = 420,
                  space_before: int = 0, space_after: int = 0) -> str:
    bold_xml = "<w:b/>" if bold else ""
    indent_xml = f'<w:ind w:firstLine="{first_line}"/>' if first_line else ""
    return (
        "<w:p>"
        "<w:pPr>"
        f'<w:jc w:val="{align}"/>'
        f"{indent_xml}"
        f'<w:spacing w:before="{space_before}" w:after="{space_after}" w:line="360" w:lineRule="auto"/>'
        "</w:pPr>"
        "<w:r>"
        "<w:rPr>"
        f"{bold_xml}"
        f'<w:rFonts w:ascii="{latin}" w:hAnsi="{latin}" w:eastAsia="{font}"/>'
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{xml_text(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def table_xml(headers, rows) -> str:
    col_width = int(9200 / len(headers))
    grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in headers)
    border = (
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        "</w:tblBorders>"
    )

    def cell(text: str, bold: bool = False) -> str:
        bold_xml = "<w:b/>" if bold else ""
        return (
            "<w:tc>"
            f'<w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
            "<w:p>"
            '<w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto"/></w:pPr>'
            "<w:r><w:rPr>"
            f"{bold_xml}"
            '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>'
            '<w:sz w:val="21"/><w:szCs w:val="21"/>'
            "</w:rPr>"
            f'<w:t xml:space="preserve">{xml_text(text)}</w:t>'
            "</w:r></w:p></w:tc>"
        )

    header_row = "<w:tr>" + "".join(cell(h, bold=True) for h in headers) + "</w:tr>"
    body_rows = "".join("<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>" for row in rows)
    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        f"{border}"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{header_row}{body_rows}"
        "</w:tbl>"
    )


def image_xml(rel_id: str, image_name: str, width_px: int, height_px: int, width_cm: float, docpr_id: int) -> str:
    width_emu = int(width_cm / 2.54 * 914400)
    height_emu = int(width_emu * height_px / width_px)
    return (
        "<w:p>"
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="40" w:line="300" w:lineRule="auto"/></w:pPr>'
        "<w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        f'<wp:docPr id="{docpr_id}" name="{escape(image_name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        "<a:graphic>"
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:pic>"
        "<pic:nvPicPr>"
        f'<pic:cNvPr id="{docpr_id}" name="{escape(image_name)}"/>'
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f'<a:blip r:embed="{rel_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing></w:r></w:p>"
    )


def build_document_xml(image_map) -> str:
    body = []
    body.append(paragraph_xml(TITLE_CN, align="center", font="黑体", size=36, bold=True, first_line=0, space_after=120))
    body.append(paragraph_xml(TITLE_EN, align="center", font="Times New Roman", latin="Times New Roman", size=24, bold=True, first_line=0, space_after=60))
    body.append(paragraph_xml(AUTHOR_LINE, align="center", size=22, first_line=0, space_after=120))
    body.append(paragraph_xml("摘要：", align="left", font="黑体", size=24, bold=True, first_line=0, space_after=30))
    body.append(paragraph_xml(ABSTRACT_CN))
    body.append(paragraph_xml(f"关键词：{KEYWORDS_CN}", align="left", size=22, first_line=0, space_after=60))
    body.append(paragraph_xml("Abstract:", align="left", font="Times New Roman", latin="Times New Roman", size=24, bold=True, first_line=0, space_after=30))
    body.append(paragraph_xml(ABSTRACT_EN, size=22))
    body.append(paragraph_xml(f"Keywords: {KEYWORDS_EN}", align="left", size=22, first_line=0, space_after=80))

    docpr_id = 1
    for section in SECTIONS:
        body.append(paragraph_xml(section["heading"], align="left", font="黑体", size=28, bold=True, first_line=0, space_before=60, space_after=40))
        for block in section["blocks"]:
            if block["type"] == "paragraph":
                body.append(paragraph_xml(block["text"]))
            elif block["type"] == "table":
                table = TABLES[block["table_id"]]
                body.append(paragraph_xml(table["title"], align="center", size=22, bold=True, first_line=0, space_before=40, space_after=20))
                body.append(table_xml(table["headers"], table["rows"]))
                body.append(paragraph_xml("", first_line=0, space_after=40))
            elif block["type"] == "figure":
                spec = FIGURES[block["figure_id"]]
                info = image_map[spec.figure_id]
                body.append(image_xml(info["rel_id"], spec.filename, info["width_px"], info["height_px"], spec.width_cm, docpr_id))
                body.append(paragraph_xml(spec.caption, align="center", size=22, bold=True, first_line=0, space_after=40))
                docpr_id += 1

    body.append(paragraph_xml("参考文献", align="left", font="黑体", size=28, bold=True, first_line=0, space_before=60, space_after=40))
    for ref in REFERENCES:
        body.append(paragraph_xml(ref, size=22, first_line=0))

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        '<w:cols w:space="425"/>'
        '<w:docGrid w:type="lines" w:linePitch="312"/>'
        "</w:sectPr>"
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:office:office" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        "<w:body>" + "".join(body) + sect_pr + "</w:body></w:document>"
    )


def content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def document_rels_xml(image_map) -> str:
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for spec in FIGURES.values():
        rel_id = image_map[spec.figure_id]["rel_id"]
        rels.append(f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{spec.filename}"/>')
    rels.append("</Relationships>")
    return "".join(rels)


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(TITLE_CN)}</dc:title>"
        "<dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex</Application><Company>OpenAI</Company><AppVersion>1.0</AppVersion></Properties>"
    )


def build_markdown():
    lines = [
        f"# {TITLE_CN}",
        "",
        f"**{TITLE_EN}**",
        "",
        AUTHOR_LINE,
        "",
        "## 摘要",
        "",
        ABSTRACT_CN,
        "",
        f"**关键词：** {KEYWORDS_CN}",
        "",
        "## Abstract",
        "",
        ABSTRACT_EN,
        "",
        f"**Keywords:** {KEYWORDS_EN}",
        "",
    ]

    for section in SECTIONS:
        lines.append(f"## {section['heading']}")
        lines.append("")
        for block in section["blocks"]:
            if block["type"] == "paragraph":
                lines.append(block["text"].replace("\n", "\n\n"))
                lines.append("")
            elif block["type"] == "table":
                table = TABLES[block["table_id"]]
                lines.append(f"**{table['title']}**")
                lines.append("")
                lines.append("| " + " | ".join(table["headers"]) + " |")
                lines.append("| " + " | ".join(["---"] * len(table["headers"])) + " |")
                for row in table["rows"]:
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
            elif block["type"] == "figure":
                spec = FIGURES[block["figure_id"]]
                lines.append(f"![{spec.caption}](figures/{spec.filename})")
                lines.append("")
                lines.append(f"*{spec.caption}*")
                lines.append("")

    lines.append("## 参考文献")
    lines.append("")
    lines.extend(REFERENCES)
    lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_docx():
    image_map = {}
    for index, spec in enumerate(FIGURES.values(), start=1):
        img_path = FIG_DIR / spec.filename
        with Image.open(img_path) as img:
            width_px, height_px = img.size
        image_map[spec.figure_id] = {
            "path": img_path,
            "width_px": width_px,
            "height_px": height_px,
            "rel_id": f"rId{index + 1}",
        }

    with zipfile.ZipFile(DOCX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("word/document.xml", build_document_xml(image_map))
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml(image_map))
        for spec in FIGURES.values():
            zf.write(FIG_DIR / spec.filename, f"word/media/{spec.filename}")


def main():
    generate_figures()
    build_markdown()
    build_docx()
    print(f"Markdown written to: {MD_PATH}")
    print(f"DOCX written to: {DOCX_PATH}")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
