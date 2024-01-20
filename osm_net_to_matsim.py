#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:48:31 2023

@author: leonefamily
"""

import re
import sys
import json
import momepy
import pyrosm
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from pathlib import Path
from shapely.ops import split
from typing import Union, Optional, Dict, List, Tuple, Any, Set
from shapely.geometry import (
    LineString, MultiLineString, Point, MultiPoint, Polygon
)

DEFAULT_SPEEDS = {
    'motorway': 130,
    'motorway_link': 80,
    'trunk': 110,
    'primary': 90,
    'primary_link': 70,
    'secondary': 60,
    # 'secondary_link': 50,
    # 'tertiary': 50,
    # 'tertiary_link': 50,
    'residential': 30,
    'living_street': 20
}
DEFAULT_SPEED = 50
DEFAULT_TRAM_SPEED = 50
DEFAULT_METRO_SPEED = 70
DEFAULT_RAIL_SPEED = 80

DEFAULT_LANES = {
    'motorway': 2,
    'trunk': 2,
    'primary': 2,
    # 'secondary': 1,
    # 'tertiary': 1,
    # 'residential': 1
}
DEFAULT_LANE = 1

SMOOTHNESS_SPEED_COEFFS = {
    # 'excellent': 1.0,
    'good': 0.9,
    'intermediate': 0.7,
    'bad': 0.5,
    'very_bad': 0.4,
    'horrible': 0.3,
    'very_horrible': 0.15,
    'impassable': 0
}
DEFAULT_SMOOTHNESS_SPEED_COEFF = 1

CATEGORY_CAPACITY_COEFFS = {
    'motorway': 1.1,
    'motorway_link': 1.1,
    # 'trunk': 1,
    # 'primary': 1,
    # 'primary_link': 1,
    'secondary': 0.95,
    'secondary_link': 0.95,
    'tertiary': 0.9,
    'tertiary_link': 0.9,
    'residential': 0.8,
    'living_street': 0.7
}
DEFAULT_CATEGORY_CAPACITY_COEFF = 1
MAX_LANE_CAPACITY = 1800  # common saturated flow value, vehicle each 2 seconds
DEFAULT_TRAM_CAPACITY = 180  # 3 consists a minute
DEFAULT_METRO_CAPACITY = 60  # 1 consist a minute
DEFAULT_RAIL_CAPACITY = 60  # 1 consist a minute

TO_CRS = 'epsg:5514'

EdgeWithData = Tuple[Tuple[float, float], Tuple[float, float], int, Dict[str, Any]]
NodeWithData = Tuple[Tuple[float, float], Dict[str, Any]]
Edge = Tuple[Tuple[float, float], Tuple[float, float], int]
ForbiddenTurns = AllowedTurns = Set[Tuple[Edge, Edge]]


def parse_args(
        args_list: Optional[List[str]] = None
) -> argparse.Namespace:
    if args_list is None:
        args_list = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--pbf-path',
        help='Path to OSM PBF file'
    )
    parser.add_argument(
        '-c', '--crs',
        help='Projected (not geographic) CRS of output files',
        default=TO_CRS
    )
    parser.add_argument(
        '-o', '--outer-border-poly-path',
        help=(
            'Path to a polygon'
            'Must be in the same CRS as -c/--crs option'
        )
    )
    parser.add_argument(
        '-d', '--higher-detail-poly-path',
        help=(
            'Path to a polygon, in which high detailed roads are kept. '
            'Must be in the same CRS as -c/--crs option'
        )
    )
    parser.add_argument(
        '-n', '--network-save-path',
        help='Save path for ouptut network'
    )
    parser.add_argument(
        '-l', '--lane-definitions-save-path',
        help='Save path for lane definitions'
    )
    parser.add_argument(
        '-e', '--edges-save-path',
        help='Save path for edges shapefile'
    )
    parser.add_argument(
        '-N', '--nodes-save-path',
        help='Save path for nodes shapefile'
    )
    args = parser.parse_args(args_list)
    return args


def get_road_net(
        osm: pyrosm.pyrosm.OSM,
        crs: str,
        higher_detail_poly: Optional[Polygon] = None,
        outer_border_poly: Optional[Polygon] = None,
        extract_lane_definitions: bool = True
) -> Tuple[nx.MultiDiGraph, Optional[ForbiddenTurns]]:
    """
    Extract road network from OSM and turn it into a cleaned directed graph.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    crs: str
        Name of coordinate reference system.
    higher_detail_poly : Optional[gpd.GeoDataFrame], optional
        Polygon to keep high detail of roads in. The default is None.
    outer_border_poly : Optional[gpd.GeoDataFrame], optional
        Polygon to crop net with. The default is None.

    Raises
    ------
    RuntimeError
        If no edges are found.

    Returns
    -------
    nx.MultiDiGraph

    """
    road_net = osm.get_data_by_custom_criteria(
        custom_filter={
            'area': ['yes'],
            'highway': ['cycleway',
                        'footway',
                        'path',
                        'pedestrian',
                        'steps',
                        'track',
                        'corridor',
                        'elevator',
                        'escalator',
                        'proposed',
                        'construction',
                        'bridleway',
                        'abandoned',
                        'platform',
                        'raceway'],
            'motor_vehicle': ['no'],
            'motorcar': ['no'],
            'service': ['parking', 'parking_aisle', 'private', 'emergency_access']
        },
        osm_keys_to_keep=['highway'],
        extra_attributes=['railway', 'maxspeed', 'lanes', 'smoothness', 'roundabout'],
        filter_type='exclude',
        keep_nodes=False,
        keep_relations=False
    )
    if road_net is None:
        raise RuntimeError('No road network edges found')
    road_net.to_crs(crs, inplace=True)

    if outer_border_poly is not None:
        road_net = gpd.clip(road_net, outer_border_poly, keep_geom_type=True)

    if higher_detail_poly is not None:
        road_net.drop(
            road_net[
                ~road_net.within(higher_detail_poly) &
                (road_net['highway'].isin(['residential', 'living_street']))
                ].index,
            inplace=True
        )

    road_net = fix_geometry(road_net)
    tags_to_dict(road_net)
    road_net.loc[
        road_net['junction'].isin({'roundabout', 'mini_roundabout'}),
        'oneway'
    ] = 'yes'

    rev_road_net = road_net[(road_net['oneway'] != 'yes')].copy()
    rev_road_net['geometry'] = rev_road_net['geometry'].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )
    road_net = gpd.GeoDataFrame(
        pd.concat([road_net, rev_road_net]), crs=road_net.crs
    ).reset_index(drop=True)

    road_net = road_net[
        road_net['railway'].isna() &
        (road_net['access'] != 'yes') &
        ~road_net['highway'].isin([
            'pedestrian', 'service', 'cycleway', 'unclassified', 'track',
            'footway', 'steps', 'path', 'rest_area', 'construction',
            'services', 'raceway', 'bus_stop', 'toll_gantry', 'via_ferrata',
            'crossing'
        ])
    ]
    road_net['lanes'] = road_net['lanes'].apply(get_lanes_number)
    road_net['maxspeed'] = road_net['maxspeed'].apply(
        convert_maxspeed_to_float
    )
    guess_road_parameters(road_net)

    graph = momepy.gdf_to_nx(road_net, directed=True)
    delete_islands_dir(graph)

    if extract_lane_definitions:
        forbidden_turns = get_all_forbidden_turns(osm, graph)
        graph, turnsgraph = check_restrictions_integrity(graph, forbidden_turns)
    else:
        forbidden_turns = None

    return graph, forbidden_turns


def get_tram_net(
        osm: pyrosm.pyrosm.OSM,
        crs: str,
        outer_border_poly: Optional[Polygon] = None
) -> nx.MultiDiGraph:
    """
    Extract tram network from OSM and turn them into a cleaned directed graph.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    crs: str
        Name of coordinate reference system.
    outer_border_poly : Optional[gpd.GeoDataFrame], optional
        Polygon to crop net with. The default is None.

    Returns
    -------
    nx.MultiDiGraph

    """
    tram_net = osm.get_data_by_custom_criteria(
        custom_filter={
            'railway': ['tram']
        },
        keep_nodes=False,
        keep_relations=False,
        extra_attributes=['maxspeed', 'railway', 'osmid']
    )
    if tram_net is None:
        logging.info('No tram links were found in the selection')
        return nx.MultiDiGraph()
    tram_net.to_crs(crs, inplace=True)

    if outer_border_poly is not None:
        tram_net.drop(
            tram_net[
                ~tram_net.within(outer_border_poly)
            ].index,
            inplace=True
        )

    tram_net = fix_geometry(tram_net)
    tags_to_dict(tram_net)
    tram_net['modes'] = 'tram'
    tram_net['lanes'] = 1
    tram_net['capacity'] = DEFAULT_TRAM_CAPACITY

    tram_net['maxspeed'] = tram_net['maxspeed'].apply(convert_maxspeed_to_float)
    tram_net.loc[tram_net['maxspeed'].isna(), 'maxspeed'] = DEFAULT_TRAM_SPEED

    add_rows = []
    rev_rows = []
    for i, row in tram_net.iterrows():
        if row['tags'] is not None:
            if 'railway:preferred_direction' in row['tags']:
                if row['tags']['railway:preferred_direction'] == 'backward':
                    rev_row = row.copy()
                    rev_geom = LineString(list(row['geometry'].coords)[::-1])
                    rev_row['geometry'] = rev_geom
                    rev_rows.append(rev_row)
                if row['tags']['railway:preferred_direction'] != 'forward':
                    add_row = row.copy()
                    add_geom = LineString(list(row['geometry'].coords)[::-1])
                    add_row['geometry'] = add_geom
                    add_rows.append(add_row)

    add_gdf = gpd.GeoDataFrame(add_rows, crs=tram_net.crs)
    rev_gdf = gpd.GeoDataFrame(rev_rows, crs=tram_net.crs)
    tram_net.loc[rev_gdf.index] = rev_gdf

    tram_net = gpd.GeoDataFrame(
        pd.concat([tram_net, add_gdf]), crs=tram_net.crs
    ).reset_index(drop=True)

    for i, row in tram_net.iterrows():
        if row['tags'] is not None:
            if 'railway:preferred_direction' in row['tags']:
                if row['tags']['railway:preferred_direction'] in ['forward', 'backward']:
                    tram_net.loc[i, 'oneway'] = 'yes'

    graph = momepy.gdf_to_nx(tram_net, directed=True)
    delete_islands_dir(graph)
    return graph


def get_metro_net(
        osm: pyrosm.pyrosm.OSM,
        crs: str,
        outer_border_poly: Optional[Polygon] = None
) -> nx.MultiDiGraph:
    """
    Extract metro network from OSM and turn it into a cleaned directed graph.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    crs: str
        Name of coordinate reference system.
    outer_border_poly : Optional[gpd.GeoDataFrame], optional
        Polygon to crop net with. The default is None.

    Returns
    -------
    nx.MultiDiGraph

    """
    metro_net = osm.get_data_by_custom_criteria(
        custom_filter={
            'railway': ['metro']
        },
        keep_nodes=False,
        keep_relations=False,
        extra_attributes=['maxspeed', 'railway', 'osmid']
    )
    if metro_net is None:
        return nx.MultiDiGraph()
    metro_net.to_crs(crs, inplace=True)

    if outer_border_poly is not None:
        metro_net.drop(
            metro_net[
                ~metro_net.within(outer_border_poly)
            ].index,
            inplace=True
        )

    metro_net = fix_geometry(metro_net)
    tags_to_dict(metro_net)
    metro_net['modes'] = 'metro'
    metro_net['lanes'] = 1
    metro_net['capacity'] = DEFAULT_TRAM_CAPACITY

    metro_net['maxspeed'] = metro_net['maxspeed'].apply(convert_maxspeed_to_float)
    metro_net.loc[metro_net['maxspeed'].isna(), 'maxspeed'] = DEFAULT_METRO_SPEED

    add_rows = []
    rev_rows = []
    for i, row in metro_net.iterrows():
        if row['tags'] is not None:
            if 'railway:preferred_direction' in row['tags']:
                if row['tags']['railway:preferred_direction'] == 'backward':
                    rev_row = row.copy()
                    rev_geom = LineString(list(row['geometry'].coords)[::-1])
                    rev_row['geometry'] = rev_geom
                    rev_rows.append(rev_row)
                if row['tags']['railway:preferred_direction'] != 'forward':
                    add_row = row.copy()
                    add_geom = LineString(list(row['geometry'].coords)[::-1])
                    add_row['geometry'] = add_geom
                    add_rows.append(add_row)

    add_gdf = gpd.GeoDataFrame(add_rows, crs=metro_net.crs)
    rev_gdf = gpd.GeoDataFrame(rev_rows, crs=metro_net.crs)
    metro_net.loc[rev_gdf.index] = rev_gdf

    metro_net = gpd.GeoDataFrame(
        pd.concat([metro_net, add_gdf]), crs=metro_net.crs
    ).reset_index(drop=True)

    for i, row in metro_net.iterrows():
        if row['tags'] is not None:
            if 'railway:preferred_direction' in row['tags']:
                if row['tags']['railway:preferred_direction'] in ['forward', 'backward']:
                    metro_net.loc[i, 'oneway'] = 'yes'

    graph = momepy.gdf_to_nx(metro_net, directed=True)
    delete_islands_dir(graph)
    return graph


def tags_to_dict(
        net: gpd.GeoDataFrame
):
    """
    Parse string tags in JSON format and keep None unchanged.

    Changes are made in place.

    Parameters
    ----------
    net : gpd.GeoDataFrame
        Any network GeoDataFrame.

    """
    net['tags'] = net['tags'].apply(
        lambda v: None if v is None else json.loads(v)
    )


def get_rail_net(
        osm: pyrosm.pyrosm.OSM,
        crs: str,
        outer_border_poly: Optional[Polygon] = None
) -> nx.MultiDiGraph:
    """
    Extract railways from OSM ways and turn them into a cleaned directed graph.

    Omits links marked as 'service'.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    crs: str
        Name of coordinate reference system.
    outer_border_poly : Optional[gpd.GeoDataFrame], optional
        Polygon to crop net with. The default is None.

    Returns
    -------
    nx.MultiDiGraph

    """
    rail_net = osm.get_data_by_custom_criteria(
        custom_filter={
            'railway': ['rail']
        },
        keep_relations=False,
        extra_attributes=['railway', 'service', 'maxspeed']
    )
    if rail_net is None:
        return nx.MultiDiGraph()
    rail_net.to_crs(crs, inplace=True)

    if outer_border_poly is not None:
        rail_net.drop(
            rail_net[
                ~rail_net.within(outer_border_poly)
            ].index,
            inplace=True
        )

    rail_net = rail_net[
        rail_net['service'].isna()
    ]
    rail_net = fix_geometry(rail_net)
    tags_to_dict(rail_net)
    rail_net['modes'] = 'rail'
    rail_net['capacity'] = DEFAULT_RAIL_CAPACITY
    rail_net['lanes'] = 1

    rail_net['maxspeed'] = rail_net['maxspeed'].apply(convert_maxspeed_to_float)
    rail_net.loc[rail_net['maxspeed'].isna(), 'maxspeed'] = DEFAULT_RAIL_SPEED

    rev_geoms = rail_net['geometry'].apply(
        lambda x: LineString(list(x.coords)[::-1])
    )
    rev_rail_net = rail_net.copy()
    rev_rail_net['geometry'] = rev_geoms

    rail_net = gpd.GeoDataFrame(
        pd.concat([rail_net, rev_rail_net]), crs=rail_net.crs
    ).reset_index(drop=True)

    graph = momepy.gdf_to_nx(rail_net, directed=True)
    delete_islands_dir(graph)
    return graph


def guess_road_speed(
        row: pd.Series
) -> Union[int, float]:
    """
    Estimate car roads' speeds based on OSM attributes.

    Parameters
    ----------
    row : pd.Series
        Pandas Series with OSM attributes.

    Returns
    -------
    Union[int, float]

    """
    speed = DEFAULT_SPEED
    if row['highway'] in DEFAULT_SPEEDS:
        speed = DEFAULT_SPEEDS[row['highway']]
    if row['smoothness'] in SMOOTHNESS_SPEED_COEFFS:
        speed *= SMOOTHNESS_SPEED_COEFFS[row['smoothness']]
    else:
        speed *= DEFAULT_SMOOTHNESS_SPEED_COEFF
    return speed


def guess_road_lanes(
        row: pd.Series
) -> Union[int, float]:
    """
    Estimate car roads' lane count based on OSM attributes.

    Parameters
    ----------
    row : pd.Series
        Pandas Series with OSM attributes.

    Returns
    -------
    Union[int, float]

    """
    lanes = DEFAULT_LANE
    if row['highway'] in DEFAULT_LANES:
        lanes = DEFAULT_LANES[row['highway']]
    return lanes


def deduce_bus_lanes(
        net: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Change original net lanes count and deduce bus only links from it.

    Parameters
    ----------
    net : gpd.GeoDataFrame
        OSM network with 'busway' attribute and possibly busway related tags.

    Returns
    -------
    gpd.GeoDataFrame

    """
    bus_rows = []
    for nr, row in net.iterrows():
        if row['tags'] is None:
            continue
        busway_tags = [t for t in row['tags'] if 'busway' in t]
        if busway_tags or not pd.isna(row['busway']):
            bus_rows.append(
                row.copy()
            )
    bus_net = gpd.GeoDataFrame(bus_rows, crs=net.crs)
    bus_net['lanes'] = 1
    return bus_net


def delete_islands_dir(
        graph: nx.MultiDiGraph
):
    """
    Remove nodes and edges from graph that are not strongly connected.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.

    """
    logging.info('Deleting islands')
    islands = list(nx.strongly_connected_components(graph))
    if len(islands) > 1:
        conns_num = {i: len(isle) for i, isle in enumerate(islands)}
        max_key = max(conns_num, key=conns_num.get)

        components = [
            graph.subgraph(c).copy() for c
            in nx.strongly_connected_components(graph)
        ]

        for i, g in enumerate(components):
            if i != max_key:
                graph.remove_nodes_from(g.nodes())
                graph.remove_edges_from(g.edges())
        logging.info(f"Removed {len(components)} islands")


def guess_road_parameters(
        net: gpd.GeoDataFrame
):
    """
    Estimate lanes count, maxspeed and capacity on car roads.

    Parameters
    ----------
    net : gpd.GeoDataFrame
        OSM car network.

    """
    # where lanes are known, divide by 2, if is not oneway
    net.loc[~net['lanes'].isna(), 'lanes'] = net[~net['lanes'].isna()].apply(
        lambda r:
        r['lanes'] if r['oneway'] == 'yes'
        else np.ceil(r['lanes'] / 2), axis=1
    )
    # estimate possible max. speed
    net.loc[net['maxspeed'].isna(), 'maxspeed'] = net[net['maxspeed'].isna()].apply(
        guess_road_speed, axis=1
    )
    net.loc[net['lanes'].isna(), 'lanes'] = net[net['lanes'].isna()].apply(
        guess_road_lanes, axis=1
    )
    bus_net = deduce_bus_lanes(net)
    # reduce lanes count by 1, where there are dedicated bus lanes
    net.loc[bus_net.index, 'lanes'] = net.loc[bus_net.index, 'lanes'] - 1
    # lanes count can be 1 or more
    net.loc[net['lanes'] < 1, 'lanes'] = 1
    bus_net['modes'] = 'pt'
    bus_net['capacity'] = 360  # every 10 seconds
    net['modes'] = 'car,pt'
    net['capacity'] = net.apply(guess_road_capacity, axis=1)


def convert_maxspeed_to_float(
        val: Union[str, int, float, None]
) -> Optional[float]:
    """
    Safely convert string speed to float.

    Parameters
    ----------
    val : Union[str, int, float, None]
        Speed value to be converted.

    Returns
    -------
    Optional[float]
        If couldn't convert, returns None.

    """
    if val is not None:
        if isinstance(val, (int, float)):
            return val
        if val.isdigit() or val.isdecimal():
            return float(val)


def guess_road_capacity(
        row: pd.Series
) -> Union[float, int]:
    """
    Estimate car road capacity based on speed, category, lanes and smoothness.

    Parameters
    ----------
    row : pd.Series
        Pandas Series with OSM attributes.

    Returns
    -------
    Union[float, int]

    """
    capacity = 100 * row['lanes']
    if row['maxspeed'] > 5:
        capacity = int(
            min(17.5 * row['maxspeed'] + 60, MAX_LANE_CAPACITY)
        ) * row['lanes']
    if row['highway'] in CATEGORY_CAPACITY_COEFFS:
        capacity *= CATEGORY_CAPACITY_COEFFS[row['highway']]
    else:
        capacity *= DEFAULT_CATEGORY_CAPACITY_COEFF
    return capacity


def generate_link_string(
        fr: str,
        to: str,
        attrs: Dict[str, str],
        add_attrs: Tuple[str] = ('geometry',)
) -> str:
    """
    Generate link string with attributes for MATSim.

    Geometry is exported as WKT string, if chosen. If attrs is empty or
    add_attrs is None or empty, no attributes are written.

    Parameters
    ----------
    fr : str
        Origin node.
    to : str
        Destination node.
    attrs : Dict[str, str]
        Attributes dictionary from graph.
    add_attrs : Tuple[str], optional
        Keys to write into attribute string. The default is ('geometry',).

    Returns
    -------
    str

    """
    l_id = attrs["link_id"]
    l_len = max(1, round(attrs["geometry"].length, 2))
    l_spd = attrs["freespeed"]
    attrstr = generate_attrs_string(
        {k: v for k, v in attrs.items() if k in add_attrs}
    )
    return (
        f'    <link id="{l_id}" '
        f'from="{fr}" to="{to}" '
        f'length="{l_len}" '
        f'capacity="{int(attrs["capacity"])}" '
        f'freespeed="{l_spd}" '
        f'modes="{attrs["modes"]}" permlanes="{attrs["permlanes"]}" >\n'
        f'{attrstr}'
        '    </link>\n'
    )


def generate_attrs_string(
        attrs: Dict[str, str]
) -> str:
    """
    Generate attributes string for MATSim link.

    Geometry is exported as WKT string, if exists in the dictionary. If attrs
    doesn't have elements, empty string is returned.

    Parameters
    ----------
    attrs : Dict[str, str]
        Attributes dictionary from graph.

    Returns
    -------
    str

    """
    s = '      <attributes>\n'
    for c, val in attrs.items():
        if not pd.isnull(val):
            if c == 'geometry':
                val = val.wkt
            s += f'        <attribute name="{c}" class="java.lang.String">{val}</attribute>\n'
    return s + '      </attributes>\n'


def assign_nodenums(
        graph: nx.MultiDiGraph
):
    """
    Set numbers to nodes attributes with 'nodenum' key.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Graph with OSM network.

    """
    for i, node in enumerate(graph.nodes):
        graph.nodes[node]['nodenum'] = i


def write_network(
        graph: nx.MultiDiGraph,
        outf: Union[str, Path]
):
    """
    Write network in MATSim format to disk.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.
    outf : Union[str, Path]
        Output path.

    """
    logging.info(f'Writing network {outf}')
    bigstring = ''
    bigstring += '<?xml version="1.0" encoding="utf-8"?>\n'
    bigstring += (
        '<!DOCTYPE network SYSTEM '
        '"http://matsim.org/files/dtd/network_v2.dtd">\n'
    )
    bigstring += '<network name="net">\n'

    bigstring += '<nodes>\n'
    for node in graph.nodes:
        nodenum = graph.nodes[node]['nodenum']
        bigstring += (
            f'    <node id="{nodenum}" x="{node[0]}" y="{node[1]}" />\n'
        )
    bigstring += '</nodes>\n'

    bigstring += '<links>\n'
    for j, edge in enumerate(graph.edges):
        from_node, to_node, edge_num = edge
        attrs = graph.edges[edge]
        fr = graph.nodes[from_node]['nodenum']
        to = graph.nodes[to_node]['nodenum']
        attrs['freespeed'] = get_freespeed(attrs['maxspeed'])
        try:
            attrs['permlanes'] = attrs['lanes']
        except:
            print(attrs)
            raise
        bigstring += generate_link_string(
            fr=fr,
            to=to,
            attrs=attrs,
            add_attrs=('geometry','osm_id')
        )

    bigstring += '</links>\n'
    bigstring += '</network>\n'

    with open(outf, mode='w', encoding='utf-8') as wr:
        wr.write(bigstring)


def multilinestring_to_linestring(
        mls: Union[MultiLineString, LineString]
) -> LineString:
    """
    Convert MultiLineString to LineString.

    Function assumes, that all internal segments are correctly ordered and
    the geometry is continuous.

    If LineString is passed to the function, it returns the same object.

    Parameters
    ----------
    mls : MultiLineString
        Correctly ordered valid MultiLineString.

    Returns
    -------
    LineString

    """
    if isinstance(mls, LineString):
        return mls
    geoms = list(mls.geoms)
    coords = list(itertools.chain.from_iterable(
        [list(g.coords) for g in geoms]
    ))
    coords_cleaned = []
    # avoid duplicated points
    # !!! ASSIGN DIFFERENT ID TO PARTS IF HAS HOLES
    for coord in coords:
        if coords_cleaned and coords_cleaned[-1] == coord:
            continue
        coords_cleaned.append(coord)
    new_geom = LineString(coords_cleaned)
    return new_geom


def fix_geometry(
        net: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Split geometry in places of touching.

    OSM topology rules do not oblige to have a node at every intersection to
    count as intersection, having same coordinates is enough. MATSim, however,
    requres those nodes, and the function solves this

    Parameters
    ----------
    net : gpd.GeoDataFrame
        Network with raw geometry after export using pyrosm.

    Returns
    -------
    gpd.GeoDataFrame

    """
    fixed_rows = []
    for i, row in net.iterrows():
        fixed_row = row.copy()
        new_row_geom = multilinestring_to_linestring(row['geometry'])
        # we are not interested in touches at the ends
        allowed_coords = list(new_row_geom.coords)[1:-1]
        if allowed_coords:
            if len(allowed_coords) > 1:
                allowed_touch = LineString(allowed_coords)
            else:
                allowed_touch = Point(allowed_coords[0])
            touching = net[net.touches(allowed_touch) & (net.index != i)]
            if not len(touching):
                continue
            for _, toucher in touching.iterrows():
                new_toucher_geom = multilinestring_to_linestring(toucher['geometry'])
                toucher_coords = list(new_toucher_geom.coords)
                common_coords = set(allowed_coords).intersection(set(toucher_coords))
                common_points = MultiPoint([Point(c) for c in common_coords])
                new_row_geom = MultiLineString(
                    split(new_row_geom, common_points)
                )
        fixed_row['geometry'] = new_row_geom
        fixed_rows.append(fixed_row)
    newnet = net.copy()
    fixnet = gpd.GeoDataFrame(fixed_rows, crs=net.crs)
    newnet.loc[newnet.drop(fixnet.index).index, 'geometry'] = (
        newnet.loc[newnet.drop(fixnet.index).index, 'geometry'].apply(
            multilinestring_to_linestring
        )
    )
    newnet.loc[fixnet.index] = fixnet
    newnet = newnet.explode(index_parts=False)
    return newnet


def get_lanes_number(
        lnum_str: Optional[str]
) -> Optional[float]:
    """
    Parse string lane count safely, or return None, if couldn't parse.

    Parameters
    ----------
    lnum_str : Optional[str]
        String with lane number.

    Returns
    -------
    Optional[float]

    """
    if lnum_str is None:
        return
    elif lnum_str.isdigit():
        return float(lnum_str)
    else:
        digits = re.findall(r'\d+', lnum_str)
        if digits:
            return float(digits[0])


def get_link_id(
        modes: str,
        from_node: int,
        to_node: int,
        id: int
) -> str:
    """
    Combine

    Parameters
    ----------
    modes : str
        Comma separated modes string.
    from_node : int
        Origin node number.
    to_node : int
        Destination node number.
    id : int
        OSM way ID or other identifier.

    Returns
    -------
    str

    """
    modes_list = modes.split(',')
    is_pt = any(m in ['pt', 'tram', 'metro', 'rail'] for m in modes_list)
    is_road = any(m in ['car', 'truck'] for m in modes_list)
    if is_pt and not is_road:
        return f'pt_{from_node}_{to_node}_{id}'
    return f'{from_node}_{to_node}_{id}'


def assign_link_ids(
        graph: nx.MultiDiGraph
):
    """
    Create the `link_id` attribute and assign a unique value to it.

    IDs are created based on `nodenum` attribute of nodes and `num` attribute
    of edges in the graph (must be assigned prior to running this function).
    Changes are made in place.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.

    """
    enumerate_edges(graph)
    for fnode, tnode, ekey, edata in graph.edges(keys=True, data=True):
        edata['link_id'] = get_link_id(
            modes=edata['modes'],
            from_node=graph.nodes[fnode]['nodenum'],
            to_node=graph.nodes[tnode]['nodenum'],
            id=edata['num']
        )


def get_freespeed(
        speed: Union[int, float],
        roundto: int = 2
) -> float:
    """
    Convert speed to m/s and round resulting float.

    Parameters
    ----------
    speed : Union[int, float]
        Speed in km/h.
    roundto : int, optional
        Number of digits after decimal point. The default is 2.

    Returns
    -------
    float

    """
    return max(1.0, round(speed / 3.6, roundto))


def find_edges_by_id(
        id: int,
        graph: nx.MultiDiGraph,
        id_attr: str = 'id'
) -> List[EdgeWithData]:
    """
    Find all graph edges containing the ``id`` value in ``id_attr`` attribute.

    Parameters
    ----------
    id : int
        Edge attribute value.
    graph : nx.MultiDiGraph
        OSM network graph.
    id_attr : str, optional
        Attribute's name to look for. The default is 'id'.

    Returns
    -------
    List[EdgeWithData]
        A list of edges with matching attributes values.

    """
    edges = []
    for e in graph.edges(data=True, keys=True):
        if e[-1][id_attr] == id:
            edges.append(e)
    return edges


def find_common_node_and_edges(
        fromid: int,
        toid: int,
        graph: nx.MultiDiGraph,
        id_attr: str = 'id'
) -> Optional[Tuple[NodeWithData, EdgeWithData, EdgeWithData]]:
    """
    Search for edges with provided attributes and their common node.

    None is returned, when there are no edges with the requested parameters
    or they do not have a common node.

    Parameters
    ----------
    fromid : int
        Start edge's attribute.
    toid : int
        End edge's attribute.
    graph : nx.MultiDiGraph
        OSM network graph.
    id_attr : str, optional
        Attribute's name to look for. The default is 'id'.

    Returns
    -------
    Optional[Tuple[NodeWithData, EdgeWithData, EdgeWithData]]
        None if the requirements are not met in the graph.

    """
    ies = find_edges_by_id(fromid, graph, id_attr=id_attr)
    oes = find_edges_by_id(toid, graph, id_attr=id_attr)
    if not ies or not oes:
        return

    for ie in ies:
        for oe in oes:
            if ie[1] == oe[0]:
                n_attrs = graph.nodes[oe[0]]
                return (oe[0], n_attrs), ie, oe


def get_forbidden_turns(
        osm: pyrosm.pyrosm.OSM,
        graph: nx.MultiDiGraph
) -> ForbiddenTurns:
    """
    Extract forbidden turns from the OSM format.

    Will only work, if the ``osm`` object was parsed at least once.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    graph : nx.MultiDiGraph
        OSM network graph.

    Returns
    -------
    ForbiddenTurns

    """
    restrictions = get_restrictions(osm=osm)
    forbidden_turns = set()
    to_skip = set()

    for i, restriction in restrictions.iterrows():
        fromid, toid, rtype = restriction[
            ['from', 'to', 'restriction_type']
        ].tolist()
        n_ie_oe = find_common_node_and_edges(fromid, toid, graph)
        if n_ie_oe is None:
            continue
        n, ie, oe = n_ie_oe
        if rtype == 'prohibitory':
            forbidden_turns.add((ie[:-1], oe[:-1]))
        elif rtype == 'mandatory':
            if fromid in to_skip:
                continue
            other_restrictions = restrictions[
                (restrictions['from'] == ie[-1]['id']) &
                (restrictions['restriction_type'] == 'mandatory')
            ]
            other_restrictions_targets = set(
                other_restrictions['to'].tolist()
            )
            forbidden_edges = [
                foe for foe in graph.out_edges(ie[:-1], data=True, keys=True)
                if foe[0] == ie[1]
                and foe[-1]['id'] not in other_restrictions_targets
            ]
            for foe in forbidden_edges:
                forbidden_turns.add((ie[:-1], foe[:-1]))
            to_skip.add(fromid)
    return forbidden_turns


def enumerate_edges(
        graph: nx.MultiDiGraph,
        attr: str = 'num',
        start: int = 0
):
    """
    Assign ``attr`` attribute to each edge graph with regard to their location.

    For one-way edges a unique number is assigned, whereas bidirectional edges
    (with the same set of start and end nodes) get duplicated numbers.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.
    attr : str, optional
        A name of attribute on each edge. The default is 'num'.
    start : int, optional
        The smallest number that edges will have. The default is 0.

    """
    to_skip = set()
    for i, e_key_data in enumerate(graph.edges(data=True, keys=True), start=start):
        fn, tn, ekey, edata = e_key_data
        if (fn, tn, ekey) in to_skip:
            continue
        oes = [
            (ofn, otn, oekey, oedata) for ofn, otn, oekey, oedata in
            graph.out_edges(tn, data=True, keys=True) if otn == fn and ofn == tn
        ]
        for ofn, otn, oekey, oedata in oes:
            oedata[attr] = i
            to_skip.add((ofn, otn, oekey))
        edata[attr] = i


def get_forbidden_uturns(
        graph: nx.MultiDiGraph,
        forbidden_turns: ForbiddenTurns
) -> ForbiddenTurns:
    """
    Create u-turns restrictions.

    A node with 2 out an 2 in edges with the same set of start and end nodes
    are considered u-turn.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.
    forbidden_turns : ForbiddenTurns
        Restricted movemens between edges.

    Returns
    -------
    ForbiddenTurns

    """
    forbidden_uturns = set()
    enumerate_edges(graph)
    for n in graph.nodes:
        oes = list(graph.out_edges(n, keys=True, data=True))  # out edges
        ies = list(graph.in_edges(n, keys=True, data=True))  # in edges

        # interested in roads that are bidirectional and are not dead ends
        if len(oes) == 2 and len(ies) == 2:
            out_ids = {oe[-1]['num'] for oe in oes}
            in_ids = {ie[-1]['num'] for ie in ies}
            common = in_ids.intersection(out_ids)
            if common == out_ids:
                for ie in ies:
                    # if ie[:-1] in forbidden_turns:
                    #     continue
                    oe = [oe for oe in oes if ie[0] == oe[1]][0]
                    forbidden_uturns.add((ie[:-1], oe[:-1]))
    return forbidden_uturns


def get_all_forbidden_turns(
        osm: pyrosm.pyrosm.OSM,
        graph: nx.MultiDiGraph
) -> ForbiddenTurns:
    """
    Extract OSM turn restrictions and turn them into a set of edges pairs.

    Also adds u-turns restrictions.

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.
    graph : nx.MultiDiGraph
        OSM network graph.

    Returns
    -------
    ForbiddenTurns
        Restricted movemens between edges.

    """
    forbidden_turns = get_forbidden_turns(osm, graph)
    forbidden_uturns = get_forbidden_uturns(graph, forbidden_turns)
    all_forbidden_turns = forbidden_turns.union(forbidden_uturns)
    return all_forbidden_turns


def check_restrictions_integrity(
        graph: nx.MultiDiGraph,
        forbidden_turns: ForbiddenTurns
) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """
    Check all turn restrictions and drop edges that are not accessible.

    This happens by converting a directed street network graph into a
    line graph that represents existing edges as nodes and connections
    between them (turns) as edges. Knowing the restrictions, forbidden
    turns are removed from the line graph, so the nodes (former edges),
    that are not accessible from at least one direction are dropped as well.
    A refined copy of initial graph is returned afterwards.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.
    forbidden_turns : ForbiddenTurns
        Restricted movemens between edges.

    Returns
    -------
    Tuple[nx.MultiDiGraph, nx.MultiDiGraph]
        A network graph cleaned according to forbidden turns.
        A graph of all possible turns within the network graph.

    """
    linegraph = nx.line_graph(graph)
    remove_turns = set()

    for e_ekey in linegraph.edges(keys=True):
        e = e_ekey[:2]
        if e in forbidden_turns:
            remove_turns.add(e_ekey)
        else:
            # only for linegraph, not essential, can be removed
            origeattrs1 = graph.edges[e[0][0], e[0][1], e[0][2]]
            try:
                opposattrs1 = graph.edges[e[0][1], e[0][0], e[0][2]]
            except:
                opposattrs1 = {'id': 'None'}
            orige1 = origeattrs1['geometry']
            origeattrs2 = graph.edges[e[1][0], e[1][1], e[1][2]]
            try:
                opposattrs2 = graph.edges[e[1][1], e[1][0], e[0][2]]
            except:
                opposattrs2 = {'id': 'None'}
            orige2 = origeattrs2['geometry']
            nodegeom1 = orige1.interpolate(orige1.length / 2)
            nodegeom2 = orige2.interpolate(orige2.length / 2)
            nx.set_edge_attributes(
                linegraph, {
                    e_ekey: {
                        'geometry': LineString([nodegeom1, nodegeom2]),
                        'edgename': str(e_ekey),
                        'id1': origeattrs1['id'],
                        'id2': origeattrs2['id'],
                    }
                }
            )
            nodeattrs1 = linegraph.nodes[e[0]]
            nodeattrs2 = linegraph.nodes[e[1]]
            if 'geometry' not in nodeattrs1:
                nodeattrs1['id'] = origeattrs1['id']
                nodeattrs1['id_opp'] = opposattrs1['id']
                nodeattrs1['geometry'] = nodegeom1
            if 'geometry' not in nodeattrs2:
                nodeattrs2['geometry'] = nodegeom2
                nodeattrs2['id'] = origeattrs2['id']
                nodeattrs2['id_opp'] = opposattrs2['id']

    for rfrom, rto, rkey in remove_turns:
        linegraph.remove_edge(rfrom, rto, key=rkey)

    cmps = list(nx.strongly_connected_components(linegraph))
    max_cmp_idx = cmps.index(max(cmps, key=len))
    linegraph = linegraph.subgraph(cmps.pop(max_cmp_idx))

    newgraph = graph.copy()
    for e_ekey in itertools.chain.from_iterable(cmps):
        newgraph.remove_edge(*e_ekey)

    new_linegraph = linegraph.copy()
    linenodes_mapping = {}
    for n, ndata in new_linegraph.nodes(data=True):
        ipt = ndata['geometry']
        linenodes_mapping[n] = ipt.x, ipt.y
    turnsgraph = nx.relabel_nodes(new_linegraph, linenodes_mapping)
    return newgraph, turnsgraph


def get_members(
        members_raw: Dict[str, np.ndarray],
        as_json_string: bool = False
) -> Union[str, List[Dict[str, Union[int, str]]]]:
    """
    Process members of relations from raw data.

    Parameters
    ----------
    members_raw : Dict[str, np.ndarray]
        Raw members dictionary from ``pyrosm`` module.
    as_json_string : bool, optional
        Whether to return as JSON string rather than list of objects.
        The default is False.

    Returns
    -------
    Union[str, List[Dict[str, Union[int, str]]]]

    """
    members = []
    for n, member_id in enumerate(members_raw['member_id']):
        members.append({
            # converting to int, because json module cannot serialize `np.int`s
            'member_id': int(member_id),
            # member_type is bytes for some reason, but member_role is not
            'member_type': members_raw['member_type'][n].decode('utf-8'),
            'member_role': members_raw['member_role'][n]
        })
    if as_json_string:
        return json.dumps(members)
    return members


def get_restrictions(
        osm: pyrosm.pyrosm.OSM
) -> pd.DataFrame:
    """
    Extract turn restrictions.

    Returns data frame containing 3 columns:
        - `restriction_type`: is either `prohibitory` (starts with "no_")
          or `mandatory` (starts with "only_"), more precise marking is not
          necessary for MATSim;
        - `from`: way ID, that is marked as initial part;
        - `to`: way ID, that is marked as end part;

    Parameters
    ----------
    osm : pyrosm.pyrosm.OSM
        OSM object of pyrosm module with a linked PBF file.

    Returns
    -------
    pd.DataFrame

    """
    relations = pd.DataFrame(osm._relations[0])
    restrictions_types = relations['tags'].apply(
        lambda x: x['restriction']
        if 'type' in x and x['type'] == 'restriction' and 'restriction' in x
        else None
    )
    relations['restriction'] = restrictions_types
    restrictions = relations[~restrictions_types.isna()].copy().reset_index(drop=True)
    restrictions.loc[
        restrictions['restriction'].str.startswith('no_'), 'restriction_type'
    ] = 'prohibitory'
    restrictions.loc[
        restrictions['restriction_type'].isna(), 'restriction_type'
    ] = 'mandatory'

    restrictions['members'] = restrictions['members'].apply(get_members)
    froms_tos = restrictions['members'].apply(
        lambda x: {member['member_role']: member['member_id'] for member in x
                   if member['member_role'] in ['from', 'to']}
    ).tolist()
    restrictions = restrictions.join(pd.DataFrame(froms_tos))
    restrictions = restrictions[['restriction_type', 'from', 'to']].dropna().astype(
        {'from': int, 'to': int}
    )
    return restrictions


def rebuild_forbidden_turns(
        graph: nx.MultiDiGraph,
        forbidden_turns: ForbiddenTurns
) -> ForbiddenTurns:
    """
    Recreate forbidden turns for a graph with removed links or nodes.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        DESCRIPTION.
    forbidden_turns : ForbiddenTurns
        DESCRIPTION.

    Returns
    -------
    ForbiddenTurns
        DESCRIPTION.

    """
    new_forbidden_turns = set()
    for ie, oe in forbidden_turns:
        try:
            # find the same edges pair on new graph
            in_edges = [
                e for e in graph.in_edges(ie[1], keys=True, data=True)
                if e[:2] == ie[:2]
            ]
            out_edges = [
                e for e in graph.out_edges(oe[0], keys=True, data=True)
                if e[:2] == oe[:2]
            ]
            if not in_edges or not out_edges:
                continue
            for iedge in in_edges:
                for oedge in out_edges:
                    new_forbidden_turns.add((iedge[:-1], oedge[:-1]))
        except (KeyError, IndexError):  # skip if there are no nodes or edges
            continue
    return new_forbidden_turns


def write_shps(
        graph: nx.MultiDiGraph,
        edges_save_path: Union[str, Path],
        nodes_save_path: Union[str, Path]
):
    """
    Write Esri Shapefiles of nodes and edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        OSM network graph.
    edges_save_path : Union[str, Path]
        Path to save edges.
    nodes_save_path : Union[str, Path]
        Path to save nodes.

    """
    nodes, edges = momepy.nx_to_gdf(graph, points=True, lines=True)
    edges.dropna(how='all', axis=1).to_file(edges_save_path, encoding='utf-8')
    nodes.dropna(how='all', axis=1).to_file(nodes_save_path, encoding='utf-8')


def get_allowed_turns(
        graph: nx.MultiDiGraph,
        forbidden_turns: ForbiddenTurns
) -> AllowedTurns:
    """
    Create allowed turns set based on forbidden ones.

    Only turns from affected links are added. Links that don't have any
    restrictions imposed are omitted.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        DESCRIPTION.
    forbidden_turns : ForbiddenTurns
        DESCRIPTION.

    Returns
    -------
    AllowedTurns

    """
    allowed_turns = set()
    to_skip = set()
    for fie, foe in forbidden_turns:
        if fie in to_skip:
            continue
        edge_forbidden_targets = {
            oe for ie, oe in forbidden_turns if fie == ie
        }
        for aoe in graph.out_edges(fie[1], keys=True):
            if aoe not in edge_forbidden_targets:
                allowed_turns.add((fie, aoe))
        to_skip.add(fie)
    return allowed_turns


def _ws(
        num: int
) -> str:
    """Get ``num`` whitespaces multiplied by 4"""
    return ' ' * num * 4


def get_lane_definitions(
        graph: nx.MultiDiGraph,
        allowed_turns: AllowedTurns
) -> str:
    ld = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    ld += (
        '<laneDefinitions xsi:schemaLocation='
        '"http://www.matsim.org/files/dtd '
        'http://www.matsim.org/files/dtd/laneDefinitions_v2.0.xsd" '
        'xmlns="http://www.matsim.org/files/dtd" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        )
    to_skip = set()
    for aie, _ in allowed_turns:
        if aie in to_skip:
            continue
        ilink_id = graph.edges[aie]['link_id']
        edge_allowed_targets = {
            oe for ie, oe in allowed_turns if aie == ie
        }
        idata = graph.edges[aie]
        fromlink = idata["link_id"]
        fromlength = idata['mm_len']
        ld += f'{_ws(1)}<lanesToLinkAssignment linkIdRef="{fromlink}">\n'
        lanecount = int(idata['lanes'])
        base_capacity = round(idata["capacity"] / lanecount)
        for lane in range(lanecount):
            fromlaneref = f'{fromlink}.{lane}'
            represented_lanes = 1  # for future
            lane_capacity = round(base_capacity * represented_lanes)
            ld += f'{_ws(2)}<lane id="{fromlaneref}">\n'
            ld += f'{_ws(3)}<leadsTo>\n'
            for aoe in edge_allowed_targets:
                odata = graph.edges[aoe]
                tolink = odata['link_id']
                ld += f'{_ws(4)}<toLink refId="{tolink}"/>\n'
            ld += f'{_ws(3)}</leadsTo>\n'
            ld += f'{_ws(3)}<representedLanes number="1"/>\n'
            ld += f'{_ws(3)}<capacity vehiclesPerHour="{lane_capacity}"/>\n'
            ld += f'{_ws(3)}<startsAt meterFromLinkEnd="{fromlength}"/>\n'
            ld += f'{_ws(3)}<alignment>{lane + 1}</alignment>\n'
            ld += f'{_ws(2)}</lane>\n'
        ld += f'{_ws(1)}</lanesToLinkAssignment>\n'
        to_skip.add(aie)
    ld += '</laneDefinitions>'
    return ld


def write_lane_definitions(
        ld: str,
        outpath: Union[str, Path]
):
    with open(outpath, mode='w', encoding='utf-8') as f:
        f.write(ld)


def main(
        pbf_path: Union[str, Path],
        crs: str = TO_CRS,
        higher_detail_poly_path: Optional[Union[str, Path]] = None,
        outer_border_poly_path: Optional[Union[str, Path]] = None,
        network_save_path: Optional[Union[str, Path]] = None,
        lane_definitions_save_path: Optional[Union[str, Path]] = None,
        edges_save_path: Optional[Union[str, Path]] = None,
        nodes_save_path: Optional[Union[str, Path]] = None
):
    osm = pyrosm.OSM(pbf_path)

    if higher_detail_poly_path is not None:
        higher_detail_poly_gdf = gpd.read_file(higher_detail_poly_path, rows=1)
        if higher_detail_poly_gdf.crs is not None:
            if higher_detail_poly_gdf.crs.srs.lower() != crs.lower():
                raise RuntimeError(
                    'High detail polygon does not have the same CRS '
                    'as the specified one'
                )
            higher_detail_poly = higher_detail_poly_gdf.loc[0, 'geometry']
        else:
            raise RuntimeError(
                'High detail polygon most likely does not have .prj'
            )
    else:
        higher_detail_poly = None

    if outer_border_poly_path is not None:
        outer_border_poly_gdf = gpd.read_file(outer_border_poly_path, rows=1)
        if outer_border_poly_gdf.crs is not None:
            if outer_border_poly_gdf.crs.srs.lower() != crs.lower():
                raise RuntimeError(
                    'Outer border polygon does not have the same CRS '
                    'as the specified one'
                )
            outer_border_poly = outer_border_poly_gdf.loc[0, 'geometry']
        else:
            raise RuntimeError(
                'Outer border polygon most likely does not have .prj'
            )
    else:
        outer_border_poly = None

    road_graph, forbidden_turns = get_road_net(
        osm=osm,
        crs=crs,
        higher_detail_poly=higher_detail_poly,
        outer_border_poly=outer_border_poly,
        extract_lane_definitions=lane_definitions_save_path is not None
    )
    rail_graph = get_rail_net(
        osm=osm,
        crs=crs,
        outer_border_poly=outer_border_poly
    )
    tram_graph = get_tram_net(
        osm=osm,
        crs=crs,
        outer_border_poly=outer_border_poly
    )
    metro_graph = get_metro_net(
        osm=osm,
        crs=crs,
        outer_border_poly=outer_border_poly
    )
    graph = nx.compose_all(
        [road_graph, rail_graph, tram_graph, metro_graph]
    )
    assign_nodenums(graph)
    assign_link_ids(graph)
    if lane_definitions_save_path is not None:
        forbidden_turns = rebuild_forbidden_turns(
            graph=graph,
            forbidden_turns=forbidden_turns
        )
        allowed_turns = get_allowed_turns(graph, forbidden_turns)
        lane_definitions = get_lane_definitions(graph, allowed_turns)
        write_lane_definitions(lane_definitions, lane_definitions_save_path)
    write_network(graph, network_save_path)
    if edges_save_path is not None and nodes_save_path is not None:
        write_shps(graph, edges_save_path, nodes_save_path)


if __name__ == '__main__':
    args = parse_args()
    main(
        pbf_path=args.pbf_path,
        crs=args.crs,
        outer_border_poly_path=args.outer_border_poly_path,
        higher_detail_poly_path=args.higher_detail_poly_path,
        network_save_path=args.network_save_path,
        edges_save_path=args.edges_save_path,
        nodes_save_path=args.nodes_save_path,
        lane_definitions_save_path=args.lane_definitions_save_path
    )
