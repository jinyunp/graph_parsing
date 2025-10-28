from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

class ChartType(str, Enum):
    line = "line"
    bar = "bar"
    stacked_bar = "stacked_bar"
    histogram = "histogram"
    scatter = "scatter"
    area = "area"
    boxplot = "boxplot"
    violin = "violin"
    heatmap = "heatmap"
    pie = "pie"
    donut = "donut"
    timeline = "timeline"
    other = "other"

class ScaleType(str, Enum):
    linear = "linear"
    log = "log"
    categorical = "categorical"
    time = "time"
    unknown = "unknown"

class Orientation(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"
    unknown = "unknown"

class MarkerType(str, Enum):
    none = "none"
    circle = "circle"
    square = "square"
    triangle = "triangle"
    diamond = "diamond"
    cross = "cross"
    plus = "plus"
    other = "other"

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
    ticks_examples: List[str] = field(default_factory=list)

@dataclass
class LegendField:
    present: bool = False
    labels: List[str] = field(default_factory=list)
    location_hint: Optional[str] = None

@dataclass
class AnnotationItem:
    text: str
    bbox: Optional[BBox] = None

@dataclass
class SeriesStyle:
    color_name: Optional[str] = None
    style_hint: Optional[str] = None
    marker: MarkerType = MarkerType.none

@dataclass
class DataSeries:
    name: Optional[str] = None
    style: SeriesStyle = field(default_factory=SeriesStyle)
    legend_label: Optional[str] = None

@dataclass
class SubplotMeta:
    title: Optional[str] = None
    x_axis: AxisField = field(default_factory=AxisField)
    y_axis: AxisField = field(default_factory=AxisField)
    series: List[DataSeries] = field(default_factory=list)
    bbox: Optional[BBox] = None

@dataclass
class ChartMetadata:
    is_chart: bool
    chart_type: Optional[ChartType] = None
    orientation: Orientation = Orientation.unknown
    title: TitleField = field(default_factory=TitleField)
    x_axis: AxisField = field(default_factory=AxisField)
    y_axis: AxisField = field(default_factory=AxisField)
    legend: LegendField = field(default_factory=LegendField)
    subplots: List[SubplotMeta] = field(default_factory=list)
    annotations_present: bool = False
    annotations: List[AnnotationItem] = field(default_factory=list)
    data_series_count: Optional[int] = None
    table_like: bool = False
    caption_nearby: Optional[str] = None
    key_phrases: List[str] = field(default_factory=list)
    secondary_y_axis: Optional[AxisField] = None
    grid_present: Optional[bool] = None
    background_image_present: Optional[bool] = None
    quality_flags: QualityFlags = field(default_factory=QualityFlags)
    confidence: float = 0.0
    source: SourceRef = field(default_factory=SourceRef)
    extra: Dict[str, Any] = field(default_factory=dict)

def to_json_dict(meta: ChartMetadata) -> Dict[str, Any]:
    def _normalize(obj):
        if isinstance(obj, Enum):
            return obj.value
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
