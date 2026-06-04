"""GIS and remote-sensing domain adapter."""
from __future__ import annotations

from src.domain_adapters.base import DomainAdapterMetadata, StaticProfileAdapter


GEO_REMOTE_SENSING_PROFILE = {
    "extends": "general",
    "exposed_tools": [
        "code_sandbox",
    ],
    "recommended_tools": [
        "wiki_search",
        "official_source_search",
        "official_doc_fetcher",
        "paper_search",
        "web_search",
        "browser",
        "calculator",
        "code_sandbox",
    ],
    "prompt_sections": [
        "gis_rs_constraint_extraction",
        "gis_rs_data_rules",
        "gis_rs_method_validation",
        "gis_rs_risk_checklist",
    ],
    "preferred_official_domains": [
        "usgs.gov",
        "landsat.gsfc.nasa.gov",
        "nasa.gov",
        "lpdaac.usgs.gov",
        "sentinels.copernicus.eu",
        "sentiwiki.copernicus.eu",
        "esa.int",
        "copernicus.eu",
        "developers.google.com/earth-engine",
        "esa-worldcover.org",
    ],
    "evidence_checklist": [
        "Dataset capabilities and band/resolution facts must be checked against official documentation.",
        "Remote-sensing method validity should be supported by papers or authoritative technical documentation.",
        "Report sensor, spatial resolution, temporal resolution, cloud/masking constraints, and scale mismatch risks.",
        "Separate evidence-backed dataset facts from speculative workflow suggestions.",
    ],
    "output_sections": [
        "Research constraint extraction: AOI, time range, target variables, candidate sensors, method needs.",
        "Dataset candidate table: dataset, purpose, spatial resolution, temporal coverage, access path, official source.",
        "Data-method fit matrix: method, required data conditions, supported datasets, unsupported reasons, risks.",
        "GIS/RS risk checklist: cloud, CRS, scale mismatch, season consistency, mixed pixels, thermal limitations.",
    ],
}


class GeoRemoteSensingAdapter(StaticProfileAdapter):
    def __init__(self) -> None:
        super().__init__(
            metadata=DomainAdapterMetadata(
                name="geo_remote_sensing",
                display_name="GIS / Remote Sensing",
                description="Domain adapter for GIS and remote-sensing research workflows.",
                keywords=(
                    "gis",
                    "remote sensing",
                    "遥感",
                    "地理信息",
                    "landsat",
                    "sentinel",
                    "modis",
                    "gee",
                    "earth engine",
                    "ndvi",
                    "lst",
                    "land use",
                    "land cover",
                    "urban heat island",
                    "城市热岛",
                    "地表温度",
                    "土地利用",
                    "影像",
                    "波段",
                    "空间分辨率",
                ),
            ),
            profile=GEO_REMOTE_SENSING_PROFILE,
        )
