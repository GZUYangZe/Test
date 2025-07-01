'''
It is the evolution of the strategy of all nodes in the network, the change of network structure, and the evolution of node strategy

The code here is a simulation involving the policy evolution of all nodes in the network
Research object: All nodes in the network, all nodes
Research content: Network entropy; The total number of edges in the network; The distribution under the evolution of three strategies; Taking one of the nodes as an example, study the evolution of the node's strategy and visualize the network structure

Compared with Example 8-15, it retains the functionality in 8-15, with the biggest difference being the addition of mean processing in this code

是网络的全体节点的策略演变下的，网络结构的变化和节点策略的演变

此处的代码，是涉及网络全体节点策略演化的仿真
研究对象：网络全体节点，所有节点
研究内容：网络熵；网络总的边的数量；三种策略的演变下的分布；以其中某一节点为例，研究的节点的策略的演变 以及可视化网络结构

与Example8-15相比，其保留了在8-15功能，最大的区别在于本处代码增加了关于均值的处理
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from itertools import combinations

# 筛选亮度较高的颜色，避免使用黑色、深蓝等难以区分的颜色
def filter_bright_colors():
    colors = list(mcolors.CSS4_COLORS.values())
    bright_colors = [
        color for color in colors if not is_dark_color(mcolors.to_rgb(color))
    ]
    return bright_colors

# 判断颜色是否为深色，通过 RGB 转为亮度计算
def is_dark_color(rgb):
    r, g, b = rgb
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness < 0.5  # 亮度小于 0.5 认为是深色

# 可视化网络图，并根据分区为节点赋予颜色
def visualize_graph_with_partition(G, entropy, partition, title, save_name=None):
    bright_colors = filter_bright_colors()
    np.random.shuffle(bright_colors)  # 随机打乱颜色顺序
    color_map = {}
    for idx, part in enumerate(partition):
        for node in part:
            color_map[node] = bright_colors[idx % len(bright_colors)]

    pos = nx.spring_layout(G)  # 使用 spring 布局
    node_colors = [color_map[node] for node in G.nodes()]

    # 设置图像大小，避免 `tight_layout()` 出现问题
    plt.figure(figsize=(8, 6))

    # 绘制图形
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors,
        edge_color='gray', font_weight='bold', node_size=500
    )

    # 设置标题
    plt.title(f"{title}\n(Entropy = {entropy:.4f})")

    # 保存图像（如果提供了保存路径）
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')  # 使用 `bbox_inches='tight'` 来调整边界

    # 显示图形
    # 禁用 PyCharm 自动的 tight_layout，直接显示图像
    plt.show()

# 获取核心划分
def getCorePartition(G):
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

# 演化过程
def simulate_evolution(num_nodes, num_rounds, epsilon=1e-9):
    G = nx.erdos_renyi_graph(num_nodes, 0.1)

    G = nx.karate_club_graph()
    m = G.number_of_edges()
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
        print("输出cross strategy", cross_probabilities / len(cross_nodes))
        print("输出跨社区节点：", cross_nodes)

        network_entropy.append(compute_network_entropy(G))

        node_degrees.append([G.degree(n) for n in G.nodes()])
        node_5_strategy_distribution.append(node_5_probabilities)

    return (G, initial_structure_info, edge_counts, entropy_values,
            global_strategy_distribution, network_entropy, key_strategy_distribution, cross_strategy_distribution, node_degrees, node_5_strategy_distribution, node_5_degrees, node_5_entropy)

# 绘制分组均值
def plot_grouped_means(data, group_size, title, ylabel):
    means = [np.mean(data[i:i + group_size]) for i in range(0, len(data), group_size)]
    plt.figure(figsize=(8, 6))
    plt.plot(data, label='Original Data')
    plt.scatter(
        [i + group_size / 2 for i in range(0, len(data), group_size)],
        means, color='red', label='Grouped Means'
    )
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


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

# 主程序
num_nodes = 50
num_rounds = 500
(G, initial_info, edge_counts, entropy_values,
 global_strategy_distribution, network_entropy, key_strategy_distribution, cross_strategy_distribution, node_degrees, node_5_strategy_distribution, node_5_degrees, node_5_entropy) = simulate_evolution(num_nodes, num_rounds)

# 绘制初始结构
visualize_graph_with_partition(G, initial_info[1], initial_info[0], "Initial Network Partition")

# 输出前10轮次和最后10轮次的社区划分
for idx in range(10):
    partition = getCorePartition(G)
    visualize_graph_with_partition(G, structuralInformation(G, partition), partition,
                                    f"Iterations {idx + 1}")
    # 保存每次迭代的图像
    plt.savefig(f"iteration_{idx + 1}.png", bbox_inches='tight')  # 保存为PNG文件
    plt.clf()  # 清除当前图像，以防图像重叠

for idx in range(num_rounds - 10, num_rounds):
    partition = getCorePartition(G)
    visualize_graph_with_partition(G, structuralInformation(G, partition), partition,
                                    f"Iterations {idx + 1}")
    # 保存每次迭代的图像
    plt.savefig(f"iteration_{idx + 1}.png", bbox_inches='tight')  # 保存为PNG文件
    plt.clf()  # 清除当前图像，以防图像重叠
'''
# 绘制边数和熵变化，带均值
plt.figure(figsize=(8, 6))
plot_grouped_means(entropy_values, 100, "Structural Information Over Time (Grouped Means)", "Structure Entropy")
plt.savefig('6-1-structure', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plot_grouped_means(network_entropy, 100, "Network Entropy Over Time (Grouped Means)", "Structure Entropy")
plt.savefig('6-2-network', bbox_inches='tight')

plt.figure(figsize=(8, 6))
plot_grouped_means(edge_counts, 100, "Edge Count Over Time (Grouped Means)", "Edge Count")
plt.savefig('6-3-Edge', bbox_inches='tight')
'''

# 绘制结构信息图，求每100个迭代的均值
plot_grouped_means(entropy_values, 100, "Structural Information Over Time (Grouped Means)", "Structure Entropy", '6-1-structure')

# 绘制网络熵图，求每100个迭代的均值
plot_grouped_means(network_entropy, 100, "Network Entropy Over Time (Grouped Means)", "Structure Entropy", '6-2-network')

# 绘制边数图，求每100个迭代的均值
plot_grouped_means(edge_counts, 100, "Edge Count Over Time (Grouped Means)", "Edge Count", '6-3-Edge')
'''
# 绘制结构信息图
plt.figure(figsize=(8, 6))
plt.plot(entropy_values, label="Structural Information")
plt.title("Structural Information Over Time (Grouped Means)")
plt.xlabel("Iterations")
plt.ylabel("Structure Entropy")
plt.legend()
plt.savefig('6-1-structure', bbox_inches='tight')  # 保存图像
plt.show()

# 绘制网络熵图
plt.figure(figsize=(8, 6))
plt.plot(network_entropy, label="Network Entropy")
plt.title("Network Entropy Over Time (Grouped Means)")
plt.xlabel("Iterations")
plt.ylabel("Structure Entropy")
plt.legend()
plt.savefig('6-2-network', bbox_inches='tight')  # 保存图像
plt.show()

# 绘制边数图
plt.figure(figsize=(8, 6))
plt.plot(edge_counts, label="Edge Count")
plt.title("Edge Count Over Time (Grouped Means)")
plt.xlabel("Iterations")
plt.ylabel("Edge Count")
plt.legend()
plt.savefig('6-3-Edge', bbox_inches='tight')  # 保存图像
plt.show()
'''
# 绘制策略分布
global_strategy_distribution = np.array(global_strategy_distribution)
key_strategy_distribution = np.array(key_strategy_distribution)
cross_strategy_distribution = np.array(cross_strategy_distribution)

plt.figure(figsize=(8, 6))
plt.plot(range(num_rounds), [dist[0] for dist in global_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in global_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in global_strategy_distribution], label="Strategy 0", color='green')
plt.title("Global Strategy Distribution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.savefig('6-4-global')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(num_rounds), [dist[0] for dist in key_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in key_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in key_strategy_distribution], label="Strategy 0", color='green')
plt.title("Inner Community Node Strategy Evolution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.savefig('6-5-inner')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(num_rounds), [dist[0] for dist in cross_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in cross_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in cross_strategy_distribution], label="Strategy 0", color='green')
plt.title("Cross Community Node Strategy Evolution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.savefig('6-6-cross')
plt.show()



# 所有节点度分布
plt.figure(figsize=(8, 6))
for i, node_degree in enumerate(np.array(node_degrees).T[:5]):  # 绘制前5个节点的度变化
    plt.plot(node_degree, label=f"Node {i} Degree")
plt.title("Node Degrees Over Time")
plt.xlabel("Iterations")
plt.ylabel("Degree")
plt.legend()
plt.savefig('6-7-node degree')
plt.show()

# 节点 5 的策略概率分布
plt.figure(figsize=(8, 6))
node_5_strategy_distribution = np.array(node_5_strategy_distribution)
plt.plot(range(num_rounds), [dist[0] for dist in node_5_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in node_5_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in node_5_strategy_distribution], label="Strategy 0", color='green')
plt.title("Node 5 Strategy Distribution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()
plt.savefig('6-8-node 5')
plt.show()

# 节点 5 的度和节点熵变化
plt.figure(figsize=(8, 6))
plt.plot(node_5_degrees, label="Node 5 Degree")
plt.plot(node_5_entropy, label="Node 5 Entropy", linestyle="--")
plt.title("Node 5 Degree and Entropy")
plt.xlabel("Iterations")
plt.ylabel("Value")
plt.legend()
plt.savefig('6-9-node 5 entropy')
plt.show()

# 以下代码都是从Example8-14的代码处复制粘贴来的=============================================================
# 单独绘制每个图========================================================================================
# 全局策略分布==========================================================================================
plt.figure()
strategy_distribution = np.array(global_strategy_distribution)


# 绘图
plt.figure(figsize=(14, 10))
# 全局策略概率分布
plt.subplot(3, 2, 1)
strategy_distribution = np.array(strategy_distribution)

plt.plot(range(num_rounds), [dist[0] for dist in strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in strategy_distribution], label="Strategy 0", color='green')
plt.title("Global Strategy Distribution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()

# 网络熵
plt.subplot(3, 2, 2)
plt.plot(network_entropy, label="Network Entropy")
plt.title("Network Entropy Over Time")
plt.xlabel("Iterations")
plt.ylabel("Entropy")
plt.legend()

# 总边数
plt.subplot(3, 2, 3)
plt.plot(edge_counts, label="Total Edges")
plt.title("Total Edges Over Time")
plt.xlabel("Iterations")
plt.ylabel("Number of Edges")
plt.legend()

# 所有节点度分布
plt.subplot(3, 2, 4)
for i, node_degree in enumerate(np.array(node_degrees).T[:5]):  # 绘制前5个节点的度变化
    plt.plot(node_degree, label=f"Node {i} Degree")
plt.title("Node Degrees Over Time")
plt.xlabel("Iterations")
plt.ylabel("Degree")
plt.legend()

# 节点 5 的策略概率分布
plt.subplot(3, 2, 5)
node_5_strategy_distribution = np.array(node_5_strategy_distribution)
plt.plot(range(num_rounds), [dist[0] for dist in node_5_strategy_distribution], label="Strategy +1", color='blue')
plt.plot(range(num_rounds), [dist[1] for dist in node_5_strategy_distribution], label="Strategy -1", color='red')
plt.plot(range(num_rounds), [dist[2] for dist in node_5_strategy_distribution], label="Strategy 0", color='green')
plt.title("Node 5 Strategy Distribution")
plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.legend()

# 节点 5 的度和节点熵变化
plt.subplot(3, 2, 6)
plt.plot(node_5_degrees, label="Node 5 Degree")
plt.plot(node_5_entropy, label="Node 5 Entropy", linestyle="--")
plt.title("Node 5 Degree and Entropy")
plt.xlabel("Rounds")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()
