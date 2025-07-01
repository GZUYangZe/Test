'''
The final major experiment in the paper
Four images are grouped together, namely:
① Structural information evolution diagram;
② Network entropy evolution diagram;
③ Evolution diagram of total number of edges;
④ Node/Cross Community Node, Strategy Evolution Diagram.

论文中的最后一个大实验
四个图为一组，分别是：
①：结构信息演变图；
②：网络熵演变图；
③：总边数演变图；
④：节点/跨社区节点，策略演变图。
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from itertools import combinations

# 筛选亮度较高的颜色
def filter_bright_colors():
    colors = list(mcolors.CSS4_COLORS.values())
    return [c for c in colors if not is_dark_color(mcolors.to_rgb(c))]

# 判断颜色是否为深色
def is_dark_color(rgb):
    r, g, b = rgb
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness < 0.5

# 可视化网络
def visualize_graph_with_partition(G, entropy, partition, title, save_name=None):
    bright_colors = filter_bright_colors()
    np.random.shuffle(bright_colors)
    color_map = {}
    for idx, part in enumerate(partition):
        for node in part:
            color_map[node] = bright_colors[idx % len(bright_colors)]
    pos = nx.spring_layout(G)
    node_colors = [color_map[node] for node in G.nodes()]
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
        font_weight='bold', node_size=500
    )
    plt.title(f"{title}\n(Entropy = {entropy:.4f})")
    # 保存图像（如果提供了保存路径）
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')  # 使用 `bbox_inches='tight'` 来调整边界

    # 显示图形
    # 禁用 PyCharm 自动的 tight_layout，直接显示图像
    plt.show()

# 划分社区并获取跨社区节点
def get_community_and_cross_nodes(G):
    partition = list(nx.community.greedy_modularity_communities(G))
    cross_nodes = set()
    for u, v in G.edges():
        for i, community in enumerate(partition):
            if u in community and v not in community:
                cross_nodes.add(u)
                cross_nodes.add(v)
    return partition, list(cross_nodes)

# 获取核心划分
def getCorePartition(G):
    G.remove_edges_from(nx.selfloop_edges(G))

    delta = max(nx.core_number(G).values())
    corenodes = []
    for i in range(delta):
        corenodes.append(list(nx.k_core(G, i + 1).nodes()))
    partition = corenodes
    for i in range(0, delta - 1):
        partition[i] = list(set(partition[i]).difference(set(partition[i + 1])))
    core_partition = []
    for i in range(delta):
        core_subgraph = nx.connected_components(
            nx.induced_subgraph(G, partition[i]))
        for sg in core_subgraph:
            core_partition.append(list(sg))
    return core_partition

# 计算结构信息熵
def structural_information_entropy(G, partition):
    G_volume = nx.volume(G, nx.nodes(G))
    entropy = 0
    for community in partition:
        v = nx.volume(G, community)
        if v == 0:
            continue
        community_entropy = 0
        for node in community:
            p = nx.degree(G, node) / v
            community_entropy += -p * math.log(p, 2) if p > 0 else 0
        entropy += (v / G_volume) * community_entropy
    return entropy
'''
之前版本的结构信息
def structuralInformation(G, P):
    G_volume = nx.volume(G, nx.nodes(G))
    entropy = 0
    all_nodes = list(nx.nodes(G))
    for item in P:
        v = nx.volume(G, item)
        g = nx.cut_size(G, item, list(set(all_nodes).difference(set(item))))
        v_entropy = 0
        for it in item:
            it_p = nx.degree(G, it) / v
            v_entropy += -it_p * math.log(it_p, 2)
        entropy += v / G_volume * v_entropy - g / G_volume * math.log(v / G_volume, 2)
    return entropy
'''
def structuralInformation(G, P):
    G_volume = nx.volume(G, nx.nodes(G))
    entropy = 0
    all_nodes = list(nx.nodes(G))
    for item in P:
        v = nx.volume(G, item)

        # 检查 v 是否为零，如果是零，则跳过该社区
        if v == 0:
            continue  # 或者你可以选择其他方式处理，比如将该社区的熵值设为零

        g = nx.cut_size(G, item, list(set(all_nodes).difference(set(item))))
        v_entropy = 0
        for it in item:
            it_p = nx.degree(G, it) / v  # 这里的除法操作之前已经确保v不为零
            v_entropy += -it_p * math.log(it_p, 2)
        entropy += v / G_volume * v_entropy - g / G_volume * math.log(v / G_volume, 2)

    return entropy

def compute_network_entropy(G):
    """计算网络熵"""
    degrees = np.array([G.degree(n) for n in G.nodes()])
    probabilities = degrees / degrees.sum()
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# 更新策略和网络
def update_strategy_and_network(G, node_index, m, epsilon=1e-9):
    """计算节点i的效用矩阵，符合给定的矩阵结构"""
    d_i = G.degree(node_index)  # 节点i的度

    neighbors = list(G.neighbors(node_index))  # 获取当前节点的邻居
    total_utility = np.zeros(3)  # 存储合作、背叛、观察的效用

    for neighbor in neighbors:
        d_j = G.degree(neighbor)  # 节点j的度
        for strategy_index in range(3):
            if strategy_index == 0:  # 合作 (C)
                # 防止度为0时计算log2
                if d_i + 1 == 0 or (2 * m + 2 + 2) == 0:
                    total_utility[strategy_index] += 0
                else:
                    total_utility[strategy_index] += -(d_i + 1) / (2 * m + 2 + 2) * np.log2((d_i + 1) / (2 * m + 2 + 2))
            elif strategy_index == 1:  # 背叛 (D)
                # 防止度为0时计算log2
                if d_i == 0 or (2 * m + 2) == 0:
                    total_utility[strategy_index] += 0
                else:
                    total_utility[strategy_index] += -(d_i) / (2 * m + 2) * np.log2(d_i / (2 * m + 2))
            else:  # 观察 (O)
                # 防止度为0时计算log2
                if d_i - 1 == 0 or (2 * m) == 0:
                    total_utility[strategy_index] += 0
                else:
                    total_utility[strategy_index] += -(d_i - 1) / (2 * m) * np.log2((d_i - 1) / (2 * m))

    # payoff_matrix = np.zeros((3, 3))  # 初始化3x3的支付矩阵

    # 计算节点的效用（注意此处修改了对payoff_matrix的使用）
    utility = total_utility

    # 确保utility是1维数组
    utility = np.array(utility)

    # 修正：直接对一维数组应用np.nanmax
    max_utility = np.nanmax(utility)

    # 计算策略选择的概率
    exp_utilities = np.exp(utility - max_utility)
    probabilities = exp_utilities / np.sum(exp_utilities)  # 归一化处理

    if probabilities.size != 3:
        raise ValueError(f"Probability size mismatch: expected 3, got {probabilities.size}")

    # 选择策略并更新网络
    chosen_strategy = np.random.choice([0, 1, 2], p=probabilities)
    delta_d = [1, 0, -1][chosen_strategy]  # 根据策略调整度
    neighbors = list(G.neighbors(node_index))
    if delta_d == 1:  # 新增边
        potential_neighbors = set(G.nodes()) - set(neighbors) - {node_index}
        if potential_neighbors:
            new_neighbor = np.random.choice(list(potential_neighbors))
            G.add_edge(node_index, new_neighbor)
    elif delta_d == -1 and neighbors:  # 删除边
        removed_neighbor = np.random.choice(neighbors)
        G.remove_edge(node_index, removed_neighbor)

    # 更新总的边数
    m = G.number_of_edges()

    return probabilities, m

# 仿真主程序
def simulate_evolution_Cross(G, num_nodes, num_rounds):
    # G = nx.erdos_renyi_graph(num_nodes, 0.1)

    # G = nx.karate_club_graph()
    m = G.number_of_edges()
    entropy_values = []
    key_strategy_distribution = []

    network_entropy = []
    edge_counts = []

    for round_idx in range(num_rounds):
        partition, cross_nodes = get_community_and_cross_nodes(G)
        entropy = structuralInformation(G, partition)
        entropy_values.append(entropy)
        edge_counts.append(m)

        cross_probabilities = np.zeros(3)
        if len(cross_nodes) > 0:
            for node in cross_nodes:
                probabilities, m = update_strategy_and_network(G, node, m)
                cross_probabilities += probabilities
            key_strategy_distribution.append(cross_probabilities / len(cross_nodes))

            network_entropy.append(compute_network_entropy(G))

        else:
            key_strategy_distribution.append([0, 0, 0]) # 没有跨社区节点时，策略分布为 0

        #key_strategy_distribution.append(cross_probabilities / len(cross_nodes))
        print(f"Round {round_idx + 1}: Entropy = {entropy:.4f}, Cross-Community Strategies = {key_strategy_distribution[-1]}")

    return G, entropy_values, key_strategy_distribution, network_entropy, edge_counts

# 演化过程
def simulate_evolution_All(G, num_nodes, num_rounds, epsilon=1e-9):
    # G = nx.erdos_renyi_graph(num_nodes, 0.1)
    # G = nx.karate_club_graph()
    G = nx.Graph(G)

    m = G.number_of_edges()
    num_nodes = G.number_of_nodes()

    initial_structure_info = None
    edge_counts = []
    node_degrees = []
    entropy_values = []

    network_entropy = []
    global_strategy_distribution = []
    key_strategy_distribution = []
    cross_strategy_distribution = []

    node_5_strategy_distribution = []
    node_5_degrees = []
    node_5_entropy = []

    for round_idx in range(num_rounds):
        partition = getCorePartition(G)
        entropy = structuralInformation(G, partition)
        if round_idx == 0:
            initial_structure_info = (partition, entropy)
        entropy_values.append(entropy)
        edge_counts.append(m)

        key_nodes = []  # 其定义的关键节点是社区内部存在边的节点
        for community in partition:
            inter_edges = sum(1 for u, v in combinations(community, 2) if G.has_edge(u, v))
            if inter_edges > 0:
                key_nodes.extend(community)

        total_probabilities = np.zeros(3)
        key_probabilities = np.zeros(3)

        # 跨社区节点，也即本文所指的关键节点
        cross_nodes = set()  # 使用集合来避免重复节点
        # 遍历每个社区
        for i, community in enumerate(partition):
            # 遍历社区中的每个节点
            for node in community:
                # 检查该节点是否与其他社区的节点有边连接
                for j, other_community in enumerate(partition):
                    if i != j:  # 如果是不同的社区
                        # 如果节点与其他社区有边连接，则认为是跨社区节点
                        if any(G.has_edge(node, other_node) for other_node in other_community):
                            cross_nodes.add(node)
                            break  # 一旦发现该节点是跨社区节点，跳出检查其他社区的循环
        # 将集合转换为列表，以便后续使用
        cross_nodes = list(cross_nodes)
        cross_probabilities = np.zeros(3)

        for node_index in G.nodes():
            probabilities, m = update_strategy_and_network(G, node_index, m)
            total_probabilities += probabilities
            if node_index in key_nodes:
                key_probabilities += probabilities
            if node_index in cross_nodes:
                cross_probabilities += probabilities

            if node_index == 5:
                node_5_probabilities = probabilities
                node_5_degrees.append(G.degree(node_index))
                d_i = G.degree(node_index)
                print("节点5的度是", d_i)
                print("网络的总的边的数量", m)
                print("网络中零分一种方式的计算的网络的总的边的数量", G.number_of_edges())
                node_5_entropy.append(- (d_i / (2 * m)) * np.log2(d_i / (2 * m) + epsilon))

        global_strategy_distribution.append(total_probabilities / num_nodes)
        key_strategy_distribution.append(key_probabilities / len(key_nodes) if key_nodes else [0, 0, 0])
        cross_strategy_distribution.append(cross_probabilities / len(cross_nodes) if cross_nodes else [0, 0, 0])

        if len(cross_nodes) > 0:
            print("输出cross strategy", cross_probabilities / len(cross_nodes))
        else:
            print("No cross-community nodes in this round.")

        print("输出跨社区节点：", cross_nodes)

        network_entropy.append(compute_network_entropy(G))

        node_degrees.append([G.degree(n) for n in G.nodes()])
        node_5_strategy_distribution.append(node_5_probabilities)

    return (G, initial_structure_info, edge_counts, entropy_values,
            global_strategy_distribution, network_entropy, key_strategy_distribution, cross_strategy_distribution, node_degrees, node_5_strategy_distribution, node_5_degrees, node_5_entropy)


# 定义一个函数来计算每100个迭代的均值
def plot_grouped_means(data, group_size, title, ylabel, filename):
    # 计算每group_size个数据的均值
    grouped_means = [np.mean(data[i:i + group_size]) for i in range(0, len(data), group_size)]
    grouped_indices = [i * group_size + group_size // 2 for i in range(len(grouped_means))]  # 计算分组的中心点

    # 绘制数据和均值
    plt.figure(figsize=(8, 6))
    plt.plot(data, label="Original Data", alpha=0.5)  # 绘制原始数据
    plt.plot(grouped_indices, grouped_means, label=f"Grouped Mean ({group_size} Iterations)", color='red',
             linestyle='--', linewidth=2)  # 绘制均值
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')  # 保存图像
    plt.show()

# 定义绘图函数，如下所示：
def plot_grouped_name(data, group_size, title, ylabel, filename):
    data = np.array(data)  # 确保数据格式正确
    num_strategies = data.shape[1]  # 计算策略的维度数量（列数）
    colors = ['blue', 'red', 'green']  # 颜色列表

    #labels = [f"Strategy {i + 1}" for i in range(num_strategies)]  # 策略标签
    # 修改图例标签
    labels = ["Strategy +1", "Strategy -1", "Strategy 0"]

    plt.figure(figsize=(8, 6))

    # 遍历每个策略（列）
    for i in range(num_strategies):
        grouped_means = [np.mean(data[j:j + group_size, i])
                         for j in range(0, len(data), group_size) if len(data[j:j + group_size, i]) > 0]

        grouped_indices = np.arange(len(grouped_means)) * group_size + group_size // 2  # 修正索引计算

        plt.plot(grouped_indices, grouped_means, label=f"{labels[i]} Mean ({group_size} Iterations)",
                 color=colors[i], linestyle='--', linewidth=2)

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', fontsize=10, frameon=True)  # 添加图例
    plt.grid(True, linestyle='--', alpha=0.5)  # 增加网格
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# 主程序
num_nodes = 250
num_edges = 1500

num_rounds = 500

# 随机生成节点度序列（度数总和为 2*num_edges，因为每条边连接两个节点）
degree_sequence = [2] * num_edges  # 这里假设每条边连接2个节点
degree_sequence.extend([0] * (num_nodes * 2 - len(degree_sequence)))  # 补充为总节点数

G1 = nx.configuration_model(degree_sequence)

# 以下是，全部节点，演变下的网络拓扑结构演变，以及策略演变
(G, initial_info_All, edge_counts_All, entropy_values_All,
 global_strategy_distribution_All, network_entropy_All, key_strategy_distribution_All, cross_strategy_distribution_All, node_degrees_All, node_5_strategy_distribution, node_5_degrees, node_5_entropy) = simulate_evolution_All(G1, num_nodes, num_rounds)

# 以下是，跨社区节点，演变下的网络拓扑结构演变，以及策略演变
G, entropy_values_Cross, key_strategy_distribution_Cross, network_entropy_Cross, edge_counts_Cross = simulate_evolution_Cross(G1, num_nodes, num_rounds)

'''
以下是，全部节点，演变下的网络拓扑结构演变，以及策略演变
'''
# 绘制结构信息图，求每100个迭代的均值
plot_grouped_means(entropy_values_All, 100, "Structural Information Over Time (All Nodes)", "Structure Entropy", '11-1-structure')

# 绘制网络熵图，求每100个迭代的均值
plot_grouped_means(network_entropy_All, 100, "Network Entropy Over Time (All Nodes)", "Structure Entropy", '11-2-network')

# 绘制边数图，求每100个迭代的均值
plot_grouped_means(edge_counts_All, 100, "Edge Count Over Time (All Nodes)", "Edge Count", '11-3-Edge')

# 绘制策略分布
global_strategy_distribution = np.array(global_strategy_distribution_All)
key_strategy_distribution = np.array(key_strategy_distribution_All)
cross_strategy_distribution = np.array(cross_strategy_distribution_All)

plot_grouped_name(global_strategy_distribution, 100, "Strategy Evolution of All Node", "Probability", '11-4-global')



plt.figure(figsize=(8, 6))
plt.plot(range(num_rounds), [dist[0] for dist in global_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in global_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in global_strategy_distribution], label="Strategy 0", color='green')
plt.title("Strategy Evolution of All Node")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.savefig('8-4-global')
plt.show()

'''
以下是，跨社区节点，演变下的网络拓扑结构演变，以及策略演变
'''
# 绘制结构信息图，求每100个迭代的均值
plot_grouped_means(entropy_values_Cross, 100, "Structural Information Over Time (Cross-community Nodes)", "Structure Entropy", '11-5-structure information')

# 绘制网络熵图，求每100个迭代的均值
plot_grouped_means(network_entropy_Cross, 100, "Network Entropy Over Time (Cross-community Nodes)", "Structure Entropy", '11-6-network')

# 绘制边数图，求每100个迭代的均值
plot_grouped_means(edge_counts_Cross, 100, "Edges Evolution Over Time (Cross-community Nodes)", "Edge Count", '11-7-edge')

# 为了确保每个绘制的图的大小一样，我们专门特意在这里，将cross community node的策略演变，的结果可视化
plot_grouped_name(key_strategy_distribution, 100, "Strategy Evolution of Cross-community Nodes", "Probability", '11-8-cross')