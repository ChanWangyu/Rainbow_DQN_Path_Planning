from shapely.geometry import Point
import folium


class Roadmap():

    def __init__(self, dataset_str):
        # xypygamexy

        self.dataset_str = dataset_str
        self.map_props = get_map_props()

        self.lower_left = get_map_props()[dataset_str]['lower_left']
        self.upper_right = get_map_props()[dataset_str]['upper_right']

        try:  # movingpandas
            from movingpandas.geometry_utils import measure_distance_geodesic
            self.max_dis_y = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                  Point(self.upper_right[0], self.lower_left[1]))
            self.max_dis_x = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                  Point(self.lower_left[0], self.upper_right[1]))
            # print(f'max_x = {self.max_dis_x}, max_y = {self.max_dis_y}')
        except:
            # hardcode
            if dataset_str == 'manhattan':
                self.max_dis_y = 21384.41050007647
                self.max_dis_x = 9925.261760248422


    def lonlat2pygamexy(self, lon, lat):
        '''
        pygamexy.
        yx
        np.array
        '''

        x = - self.max_dis_x * (lat - self.upper_right[1]) / (self.upper_right[1] - self.lower_left[1])
        y = self.max_dis_y * (lon - self.lower_left[0]) / (self.upper_right[0] - self.lower_left[0])
        return x, y

    def pygamexy2lonlat(self, x, y):
        # print(x, y)
        lon = y * (self.upper_right[0] - self.lower_left[0]) / self.max_dis_y + self.lower_left[0]
        lat = - x * (self.upper_right[1] - self.lower_left[1]) / self.max_dis_x + self.upper_right[1]
        return lon, lat


def get_map_props():
    map_props = {
        'manhattan':
            {
                'lower_left': [-74.025000, 40.689648],  # lon, lat 40.882214, 40.689648, -73.907581, -74.025000
                'upper_right': [-73.907581, 40.882214]
            }
    }
    return map_props


def traj_to_timestamped_geojson(index, trajectory, num_uav, num_agent, color, only_UVs=False):  # indextrajindexenum

    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index  # datetimeindex
    point_gdf["previous_time"] = point_gdf["time"].shift()

    features = []
    # for Point in GeoJSON type
    for _, row in point_gdf.iterrows():
        if only_UVs and index >= num_agent: break
        corrent_point_coordinates = [row["geometry"].xy[0][0], row["geometry"].xy[1][0]]
        current_time = [row["time"].isoformat()]

        # ra = {'uav': 3, 'car': 5, 'human': 0.5}  #
        ra = {'uav': 5, 'car': 7, 'human': 2}  #
        op = {'uav': 1, 'car': 1, 'human': 1}  # opacity0.2

        if index < num_uav:  # UAV
            radius, opacity = ra['uav'], op['uav']
        elif num_uav <= index < num_agent:  # CAR
            radius, opacity = ra['car'], op['car']
        else:  # human
            radius, opacity = ra['human'], op['human']

        # for Point in GeoJSON type  (Temporally Deprecated)
        features.append(  #
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": corrent_point_coordinates,
                },
                "properties": {
                    "times": current_time,
                    "icon": 'circle',  # point
                    "iconstyle": {
                        'fillColor': color,
                        'fillOpacity': opacity,  #
                        'stroke': 'true',
                        'radius': radius,
                        'weight': 1,
                    },
                    "style": {  # line
                        "color": color,
                        "opacity": opacity
                    },
                    "code": 11,

                },
            }
        )
    return features


def folium_draw_circle(map, pos, color, radius, weight):  #
    folium.vector_layers.Circle(
        location=pos,  #
        radius=radius,  #  m
        color=color,  #
        # fill=True,  #
        # fill_color='#%02X%02X%02X' % (0, 0, 0),  #
        # fillOpacity=1,  # Fill opacity
        weight=weight  #
    ).add_to(map)


def folium_draw_CircleMarker(map, pos, color, radius):  #
    folium.CircleMarker(
        location=pos,
        radius=radius,
        color=color,
        stroke=False,
        fill=True,
        fill_opacity=1,
        opacity=1,
        popup="{} ".format(radius),
        tooltip=str(pos),
    ).add_to(map)


def get_border(ur, lf):
    upper_left = [lf[0], ur[1]]
    upper_right = [ur[0], ur[1]]
    lower_right = [ur[0], lf[1]]
    lower_left = [lf[0], lf[1]]

    coordinates = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left
    ]

    geo_json = {"type": "FeatureCollection",
                "properties": {
                    "lower_left": lower_left,
                    "upper_right": upper_right
                },
                "features": []}

    grid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }

    geo_json["features"].append(grid_feature)

    return geo_json


if __name__ == "__main__":
    from movingpandas.geometry_utils import measure_distance_geodesic

    lower_left = [-74.025000, 40.689648]
    upper_right = [-73.907581, 40.882214]
    max_dis_y = measure_distance_geodesic(Point(lower_left[0], lower_left[1]),
                                               Point(upper_right[0], lower_left[1]))
    max_dis_x = measure_distance_geodesic(Point(lower_left[0], lower_left[1]),
                                               Point(lower_left[0], upper_right[1]))
    print(max_dis_x, max_dis_y)



