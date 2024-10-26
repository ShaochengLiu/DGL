import dgl
import networkx as nx
import torch

def load_edges_file(file_path):
    """
    加载带有注释行的 edges 文件，并根据无向图/有向图的结构生成 DGL 图，处理多重边、双向边、权重和时间戳。
    支持多种数据集格式，包括字符串类型的顶点。
    
    :param file_path: edges 文件路径
    :return: DGL 图
    """
    edges = []
    weights = []
    timestamps = []
    is_directed = True  # 默认有向图
    found_comment_line = False  # 是否找到注释行
    has_weight = False  # 是否包含权重列
    has_timestamp = False  # 是否包含时间戳

    with open(file_path, 'r') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue

            # 检查是否是注释行
            if line.startswith('%'):
                found_comment_line = True
                # 检查是否有 "asym" 表示有向图
                if 'asym' in line.lower():
                    is_directed = True
                # 如果有 "sym" 表示无向图
                elif 'sym' in line.lower():
                    is_directed = False
                continue

            # 处理边信息，兼容空格或逗号分隔的格式
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split()

            # 根据数据列数判断边的类型
            if len(parts) == 2:  # 无权重无时间戳
                src, dst = parts[0], parts[1]  # 保持顶点为字符串
                edges.append((src, dst))  # 添加边
            elif len(parts) == 3:  # 带权重，无时间戳
                src, dst, weight = parts[0], parts[1], float(parts[2])
                edges.append((src, dst))  # 添加边
                weights.append(weight)  # 添加权重
                has_weight = True
            elif len(parts) == 4:  # 带权重和时间戳
                src, dst, weight, timestamp = parts[0], parts[1], float(parts[2]), float(parts[3])
                edges.append((src, dst))  # 添加边
                weights.append(weight)  # 添加权重
                timestamps.append(timestamp)  # 添加时间戳
                has_weight = True
                has_timestamp = True

            # 如果无向图，添加双向边
            if not is_directed:
                edges.append((dst, src))  # 添加双向边
                if has_weight:
                    weights.append(weight)  # 双向边的权重相同
                if has_timestamp:
                    timestamps.append(timestamp)  # 双向边的时间戳相同

    # 如果没有找到注释行，默认有向图
    if not found_comment_line:
        is_directed = True

    # 创建 networkx 图对象
    if is_directed:
        g_nx = nx.MultiDiGraph()  # 有向多重边图
    else:
        g_nx = nx.MultiDiGraph()  # 也使用 MultiDiGraph 处理无向双向边

    # 添加边到 networkx 图，允许多重边
    g_nx.add_edges_from(edges)

    # 将 networkx 图转换为 DGL 图
    g_dgl = dgl.from_networkx(g_nx)

    # 确保权重和时间戳的长度与实际边数匹配
    assert len(weights) == 0 or len(weights) == g_dgl.number_of_edges(), "权重数量和边数量不一致"
    assert len(timestamps) == 0 or len(timestamps) == g_dgl.number_of_edges(), "时间戳数量和边数量不一致"

    # 手动添加权重到 DGL 图中
    if has_weight:
        g_dgl.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    # 手动添加时间戳到 DGL 图中
    if has_timestamp:
        g_dgl.edata['timestamp'] = torch.tensor(timestamps, dtype=torch.float32)

    return g_dgl





file_path_edges = r"D:\\testDGL\\edges\\edges,产品打分图rec-amz-Sports-and-Outdoors\\rec-amz-Sports-and-Outdoors.edges"  

# 加载 edges 文件
graph_edges = load_edges_file(file_path_edges)
print(f"Edges file loaded as DGL graph: {graph_edges}")
