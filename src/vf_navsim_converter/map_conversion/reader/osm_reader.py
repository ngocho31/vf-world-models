"""Lanelet2 / OSM reader implementation."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from ..contracts.osm_models import OSMNode, OSMRelation, OSMRelationMember, OSMWay, ParsedLaneletMap


class Lanelet2OSMReader:
    """Read a Lanelet2 OSM XML file into a :class:`ParsedLaneletMap`."""

    def read(self, source_path: Path) -> ParsedLaneletMap:
        source_path = Path(source_path)
        tree = ET.parse(source_path)
        root = tree.getroot()

        parsed = ParsedLaneletMap(source_path=source_path)

        for node in root.findall("node"):
            node_id = int(node.get("id"))
            lat = float(node.get("lat"))
            lon = float(node.get("lon"))
            ele = 0.0
            ele_tag = node.find('tag[@k="ele"]')
            if ele_tag is not None and ele_tag.get("v") is not None:
                ele = float(ele_tag.get("v"))
            parsed.nodes[node_id] = OSMNode(node_id=node_id, lat=lat, lon=lon, ele=ele)

        for way in root.findall("way"):
            way_id = int(way.get("id"))
            node_refs = [int(nd.get("ref")) for nd in way.findall("nd") if nd.get("ref")]
            tags = _collect_tags(way.findall("tag"))
            parsed.ways[way_id] = OSMWay(way_id=way_id, node_refs=node_refs, tags=tags)

        for relation in root.findall("relation"):
            relation_id = int(relation.get("id"))
            members = [
                OSMRelationMember(
                    member_type=member.get("type", ""),
                    ref=int(member.get("ref")),
                    role=member.get("role", ""),
                )
                for member in relation.findall("member")
                if member.get("ref")
            ]
            tags = _collect_tags(relation.findall("tag"))
            parsed.relations[relation_id] = OSMRelation(
                relation_id=relation_id, members=members, tags=tags
            )

        return parsed


def _collect_tags(tag_nodes: List[ET.Element]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for tag in tag_nodes:
        key = tag.get("k")
        value = tag.get("v")
        if key is not None and value is not None:
            tags[key] = value
    return tags
