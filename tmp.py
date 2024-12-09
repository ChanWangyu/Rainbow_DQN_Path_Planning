import networkx as nx

# 读取 graphml 文件
graph = nx.read_graphml(f'ref/data/manhattan/undirected_partial_road_network.graphml')

# 获取节点和边的数量
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()

print(f"节点数: {num_nodes}")
print(f"边数: {num_edges}")
