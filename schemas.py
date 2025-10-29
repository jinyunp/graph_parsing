from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

class ScaleType(str, Enum):
    linear = "linear"
    log = "log"
    categorical = "categorical"
    datetime = "datetime"
    unknown = "unknown"

class Orientation(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"
    mixed = "mixed"
    unknown = "unknown"

@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class SourceRef:
    source_pdf: Optional[str] = None
    page_number: Optional[int] = None
    image_path: Optional[str] = None
    image_sha1: Optional[str] = None
    bbox: Optional[BBox] = None

@dataclass
class QualityFlags:
    low_resolution: bool = False
    cropped_or_cutoff: bool = False
    non_korean_text_present: bool = False
    heavy_watermark: bool = False
    skew_or_perspective: bool = False

@dataclass
class TitleField:
    text: Optional[str] = None
    is_inferred: bool = False

@dataclass
class AxisField:
    name: Optional[str] = None
    unit: Optional[str] = None
    is_inferred: bool = False
    scale: ScaleType = ScaleType.unknown

@dataclass
class LegendField:
    present: bool = False
    labels: List[str] = field(default_factory=list)
    location_hint: Optional[str] = None

@dataclass
class SeriesItem:
    label: Optional[str] = None
    label_is_inferred: bool = False
    sample_points: List[Any] = field(default_factory=list)
    style_hint: Optional[str] = None
    summary: Optional[str] = None

@dataclass
class SubplotMeta:
    title: Optional[str] = None
    x_axis: AxisField = field(default_factory=AxisField)
    y_axis: AxisField = field(default_factory=AxisField)
    series: List[SeriesItem] = field(default_factory=list)
    bbox: Optional[BBox] = None

@dataclass
class ChartMetadata:
    is_chart: bool
    chart_type: Optional[str] = None
    orientation: Orientation = Orientation.unknown
    title: TitleField = field(default_factory=TitleField)
    x_axis: AxisField = field(default_factory=AxisField)
    y_axis: AxisField = field(default_factory=AxisField)
    secondary_y_axis: AxisField = field(default_factory=AxisField)
    legend: LegendField = field(default_factory=LegendField)
    data_series_count: Optional[int] = None
    series: List[SeriesItem] = field(default_factory=list)
    subplots: List[SubplotMeta] = field(default_factory=list)
    annotations_present: bool = False
    annotations: List[str] = field(default_factory=list)
    table_like: bool = False
    grid_present: Optional[bool] = None
    background_image_present: Optional[bool] = None
    caption_nearby: Optional[str] = None
    quality_flags: QualityFlags = field(default_factory=QualityFlags)
    confidence: float = 0.0
    source: SourceRef = field(default_factory=SourceRef)
    key_phrases: List[str] = field(default_factory=list)

def to_json_dict(meta: ChartMetadata) -> Dict[str, Any]:
    def _normalize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            d = asdict(obj)
            for k, v in d.items():
                d[k] = _normalize(v)
            return d
        if isinstance(obj, list):
            return [_normalize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in obj.items()}
        return obj
    return _normalize(meta)
