import networkx as nx
from scipy.io import mmread
import scipy.sparse as sp
import dgl
import torch

def load_mtx_file(file_path):
    """
    加载 .mtx 格式的文件，并将其转换为 DGL 图。只有在存在权重的情况下才添加权重。
    
    :param file_path: .mtx 文件路径
    :return: DGL 图
    """
    # 读取 .mtx 文件为稀疏矩阵
    mtx_matrix = mmread(file_path)

    # 检查是否为稀疏矩阵
    if not sp.issparse(mtx_matrix):
        raise ValueError("MTX file does not contain a valid sparse matrix.")

    # 将稀疏矩阵转换为 COO 格式（行-列-值）
    coo_matrix = mtx_matrix.tocoo()

    # 提取边的源节点、目标节点
    edges = list(zip(coo_matrix.row, coo_matrix.col))

    # 确定是否有权重，通过检查 `coo_matrix` 的数据是否包含实际的权重
    weights = coo_matrix.data if coo_matrix.data is not None and coo_matrix.data.size > 0 else None
    has_weights = weights is not None and weights.size > 0

    # 根据矩阵是否对称，判断是否为无向图
    is_directed = not (coo_matrix != coo_matrix.T).nnz == 0

    # 创建 NetworkX 图对象
    if is_directed:
        g_nx = nx.DiGraph()  # 有向图
    else:
        g_nx = nx.Graph()  # 无向图

    # 添加边到 NetworkX 图
    for src, dst in edges:
        g_nx.add_edge(int(src), int(dst))

    # 将 NetworkX 图转换为 DGL 图
    g_dgl = dgl.from_networkx(g_nx)

    # 如果数据集中有权重，才将其添加到 DGL 图的 edata
    if has_weights:
        g_dgl.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return g_dgl

# 加载 mtx 文件
file_path_mtx = r"D:\\testDGL\\mtx\\2D_27628_bjtcai\\2D_27628_bjtcai.mtx"
graph_mtx = load_mtx_file(file_path_mtx)
print(f"MTX file loaded as DGL graph: {graph_mtx}")
