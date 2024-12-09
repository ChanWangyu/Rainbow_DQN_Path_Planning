import pandana as pdna
import pandas as pd
import numpy as np
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
# import shapely, geopandas, fiona
# from shapely.geometry import Point,LineString
from pandana.loaders import osm
import matplotlib.pyplot as plt
import matplotlib
import random
warnings.filterwarnings('ignore')
# E:/brucechen/python project/RL-rs
if __name__ == '__main__':

    # # net = pdna.Network(nodes["latitude"], nodes["longitude"], edges["Start"], edges["End"], edges[["Weight"]])
    # network = osm.pdna_network_from_bbox(37.698, -122.517, 37.819, -122.354)
    # net = osm.pdna_network_from_bbox(40.6, -73.7, 40.9, -74.1)
    # net.nodes_df.to_csv('../data/new_nodes.csv')
    # net.edges_df.to_csv('../data/new_edges.csv')
    # # network.nodes_df.to_csv('../data/San_nodes.csv')
    # # network.edges_df.to_csv('../data/San_edges.csv')
    # nodes = pd.read_csv('../data/San_nodes.csv', index_col=0)
    # edges = pd.read_csv('../data/San_edges.csv', index_col=[0, 1])
    New_edges = pd.read_csv('../../data/NY/new_edges.csv', index_col=[0, 1])
    New_nodes = pd.read_csv('../../data/NY/new_nodes.csv', index_col=0)
    # # network = pdna.Network(nodes['x'], nodes['y'], edges['from'], edges['to'], edges[['distance']])
    network = pdna.Network(New_nodes['x'], New_nodes['y'], New_edges['from'], New_edges['to'], New_edges[['distance']])
    print(network.nodes_df)
    plt.scatter(network.nodes_df.x, network.nodes_df.y,
                s=1, cmap='YlOrRd',
                norm=matplotlib.colors.LogNorm())
    cb = plt.colorbar()
    plt.show()
    # dis = network.shortest_path(42723127, 42914010)
    # print(dis)
    # print("finished")

    # df = pq.read_table(u'../data/yellow_tripdata_2022-01.parquet').to_pandas()

    # shp_df = geopandas.GeoDataFrame.from_file(r'../data/roads_free.shp', encoding='gb18030')
    # print(shp_df.head())
    # geopandas.GeoDataFrame.to_csv(shp_df, '../data/roads_free.csv')
    # shp_df.plot()
    # plt.show()

    # p = LineString([(-78.7663422, 42.9541481), (-78.7662288, 42.9535533), (-78.7661382, 42.9522885),
    #                 (-78.7660897, 42.9518698), (-78.7659965, 42.9515302), (-78.7658586, 42.9512508), (-78.7656671, 42.9509604)])
    # x, y=p.xy
    # plt.plot(x, y, c="red")
    # plt.show()
