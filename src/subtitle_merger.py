from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import hypot

from src.models import FrameOcrResult, OcrLine, SubtitleSegment
from src.text_filter import frame_text, line_box, line_center
from src.utils import text_key


def text_similarity(left: str, right: str) -> float:
    left_key = text_key(left)
    right_key = text_key(right)
    if not left_key and not right_key:
        return 1.0
    if not left_key or not right_key:
        return 0.0

    sequence_score = SequenceMatcher(None, left_key, right_key).ratio()
    left_lines = {text_key(line) for line in left.splitlines() if text_key(line)}
    right_lines = {text_key(line) for line in right.splitlines() if text_key(line)}
    if not left_lines or not right_lines:
        return sequence_score

    line_score = len(left_lines & right_lines) / len(left_lines | right_lines)
    return max(sequence_score, line_score)


def average_confidence(frame_result: FrameOcrResult) -> float:
    if not frame_result.lines:
        return 0.0
    return sum(line.confidence for line in frame_result.lines) / len(frame_result.lines)


def box_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) < 4 or len(right) < 4:
        return 0.0

    left_x1, left_y1, left_x2, left_y2 = [float(value) for value in left[:4]]
    right_x1, right_y1, right_x2, right_y2 = [float(value) for value in right[:4]]
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def normalized_center_distance(
    left: OcrLine,
    right: OcrLine,
    frame_width: float,
    frame_height: float,
) -> float:
    left_center = line_center(left)
    right_center = line_center(right)
    if left_center is None or right_center is None or frame_width <= 0 or frame_height <= 0:
        return 1.0

    distance = hypot(
        (left_center[0] - right_center[0]) / frame_width,
        (left_center[1] - right_center[1]) / frame_height,
    )
    return min(1.0, distance)


def estimate_frame_size(frame_results: list[FrameOcrResult]) -> tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for frame_result in frame_results:
        for line in frame_result.lines:
            box = line_box(line)
            if box is None:
                continue
            max_x = max(max_x, box[2])
            max_y = max(max_y, box[3])
    return max_x, max_y


def line_position_matches(
    left: OcrLine,
    right: OcrLine,
    frame_width: float,
    frame_height: float,
    iou_threshold: float = 0.3,
    center_distance_threshold: float = 0.08,
) -> bool:
    if box_iou(left.box, right.box) >= iou_threshold:
        return True
    return (
        normalized_center_distance(left, right, frame_width, frame_height)
        <= center_distance_threshold
    )


def line_match_score(
    left_lines: list[OcrLine],
    right_lines: list[OcrLine],
    frame_width: float,
    frame_height: float,
    text_similarity_threshold: float = 0.85,
    iou_threshold: float = 0.3,
    center_distance_threshold: float = 0.08,
) -> float:
    if not left_lines and not right_lines:
        return 1.0
    if not left_lines or not right_lines:
        return 0.0

    matched_right_indexes: set[int] = set()
    matched_count = 0
    for left_line in left_lines:
        best_index: int | None = None
        best_score = 0.0
        for index, right_line in enumerate(right_lines):
            if index in matched_right_indexes:
                continue
            score = text_similarity(left_line.text, right_line.text)
            if score < text_similarity_threshold:
                continue
            if not line_position_matches(
                left_line,
                right_line,
                frame_width=frame_width,
                frame_height=frame_height,
                iou_threshold=iou_threshold,
                center_distance_threshold=center_distance_threshold,
            ):
                continue
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            matched_right_indexes.add(best_index)
            matched_count += 1

    return matched_count / max(len(left_lines), len(right_lines))


def line_position_score(
    left: OcrLine,
    right: OcrLine,
    frame_width: float,
    frame_height: float,
) -> float:
    iou_score = box_iou(left.box, right.box)
    distance_score = 1.0 - normalized_center_distance(
        left,
        right,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return max(iou_score, distance_score)


@dataclass
class TextTrack:
    start_ms: int
    last_seen_ms: int
    line: OcrLine
    confidences: list[float] = field(default_factory=list)

    def update(self, timestamp_ms: int, line: OcrLine) -> None:
        self.last_seen_ms = timestamp_ms
        self.confidences.append(line.confidence)
        if len(text_key(line.text)) > len(text_key(self.line.text)):
            self.line = line
            return
        if len(text_key(line.text)) == len(text_key(self.line.text)) and line.confidence > self.line.confidence:
            self.line = line

    def to_segment(self, frame_interval_ms: int) -> SubtitleSegment:
        confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        return SubtitleSegment(
            start_ms=self.start_ms,
            end_ms=max(self.start_ms + frame_interval_ms, self.last_seen_ms + frame_interval_ms),
            text=self.line.text,
            confidence=confidence,
        )


def _track_line_match_score(
    track: TextTrack,
    line: OcrLine,
    frame_width: float,
    frame_height: float,
    text_similarity_threshold: float,
    iou_threshold: float,
    center_distance_threshold: float,
    use_position: bool,
) -> float:
    text_score = text_similarity(track.line.text, line.text)
    if text_score < text_similarity_threshold:
        return 0.0

    if not use_position:
        return text_score

    position_matches = line_position_matches(
        track.line,
        line,
        frame_width=frame_width,
        frame_height=frame_height,
        iou_threshold=iou_threshold,
        center_distance_threshold=center_distance_threshold,
    )
    if not position_matches:
        return 0.0

    position_score = line_position_score(
        track.line,
        line,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return text_score * 0.7 + position_score * 0.3


def merge_text_tracks_state_machine(
    frame_results: list[FrameOcrResult],
    frame_interval_ms: int,
    max_gap_ms: int | None = None,
    use_position: bool = True,
    line_text_similarity_threshold: float = 0.85,
    line_iou_threshold: float = 0.3,
    line_center_distance_threshold: float = 0.08,
) -> list[SubtitleSegment]:
    if not frame_results:
        return []

    max_gap_ms = max_gap_ms if max_gap_ms is not None else int(frame_interval_ms * 1.5)
    frame_width, frame_height = estimate_frame_size(frame_results)
    active_tracks: list[TextTrack] = []
    completed_tracks: list[TextTrack] = []

    def close_stale_tracks(timestamp_ms: int) -> None:
        still_active: list[TextTrack] = []
        for track in active_tracks:
            if timestamp_ms - track.last_seen_ms > max_gap_ms:
                completed_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks[:] = still_active

    for frame_result in sorted(frame_results, key=lambda item: item.timestamp_ms):
        close_stale_tracks(frame_result.timestamp_ms)
        lines = [line for line in frame_result.lines if text_key(line.text)]
        if not lines:
            continue

        possible_matches: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active_tracks):
            for line_index, line in enumerate(lines):
                score = _track_line_match_score(
                    track,
                    line,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    text_similarity_threshold=line_text_similarity_threshold,
                    iou_threshold=line_iou_threshold,
                    center_distance_threshold=line_center_distance_threshold,
                    use_position=use_position,
                )
                if score > 0.0:
                    possible_matches.append((score, track_index, line_index))

        matched_tracks: set[int] = set()
        matched_lines: set[int] = set()
        for _, track_index, line_index in sorted(possible_matches, reverse=True):
            if track_index in matched_tracks or line_index in matched_lines:
                continue
            active_tracks[track_index].update(frame_result.timestamp_ms, lines[line_index])
            matched_tracks.add(track_index)
            matched_lines.add(line_index)

        for line_index, line in enumerate(lines):
            if line_index in matched_lines:
                continue
            active_tracks.append(
                TextTrack(
                    start_ms=frame_result.timestamp_ms,
                    last_seen_ms=frame_result.timestamp_ms,
                    line=line,
                    confidences=[line.confidence],
                )
            )

    completed_tracks.extend(active_tracks)
    ordered_tracks = sorted(
        completed_tracks,
        key=lambda track: (
            track.start_ms,
            line_box(track.line)[1] if line_box(track.line) else 0.0,
            line_box(track.line)[0] if line_box(track.line) else 0.0,
            track.last_seen_ms,
            track.line.text,
        ),
    )
    return [
        track.to_segment(frame_interval_ms=frame_interval_ms)
        for track in ordered_tracks
    ]


def merge_repeated_text_snapshots(
    frame_results: list[FrameOcrResult],
    frame_interval_ms: int,
    similarity_threshold: float = 0.92,
    max_gap_ms: int | None = None,
    use_position: bool = True,
    position_match_threshold: float = 0.5,
    line_text_similarity_threshold: float = 0.85,
    line_iou_threshold: float = 0.3,
    line_center_distance_threshold: float = 0.08,
) -> list[SubtitleSegment]:
    if not frame_results:
        return []

    max_gap_ms = max_gap_ms if max_gap_ms is not None else int(frame_interval_ms * 1.5)
    segments: list[SubtitleSegment] = []
    current_text = ""
    current_lines: list[OcrLine] = []
    start_ms = 0
    last_seen_ms = 0
    confidences: list[float] = []

    def close_current() -> None:
        nonlocal current_text, current_lines, start_ms, last_seen_ms, confidences
        if not current_text:
            return
        end_ms = max(start_ms + frame_interval_ms, last_seen_ms + frame_interval_ms)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        segments.append(
            SubtitleSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=current_text,
                confidence=confidence,
            )
        )
        current_text = ""
        current_lines = []
        confidences = []

    frame_width, frame_height = estimate_frame_size(frame_results)

    for frame_result in sorted(frame_results, key=lambda item: item.timestamp_ms):
        text = frame_text(frame_result.lines)
        if not text:
            if current_text and frame_result.timestamp_ms - last_seen_ms > max_gap_ms:
                close_current()
            continue

        confidence = average_confidence(frame_result)
        if not current_text:
            current_text = text
            current_lines = frame_result.lines
            start_ms = frame_result.timestamp_ms
            last_seen_ms = frame_result.timestamp_ms
            confidences = [confidence]
            continue

        is_similar = text_similarity(current_text, text) >= similarity_threshold
        is_close_enough = frame_result.timestamp_ms - last_seen_ms <= max_gap_ms
        position_score = line_match_score(
            current_lines,
            frame_result.lines,
            frame_width=frame_width,
            frame_height=frame_height,
            text_similarity_threshold=line_text_similarity_threshold,
            iou_threshold=line_iou_threshold,
            center_distance_threshold=line_center_distance_threshold,
        )
        is_position_match = (not use_position) or position_score >= position_match_threshold
        if is_similar and is_close_enough and is_position_match:
            if len(text_key(text)) > len(text_key(current_text)):
                current_text = text
                current_lines = frame_result.lines
            last_seen_ms = frame_result.timestamp_ms
            confidences.append(confidence)
            continue

        close_current()
        current_text = text
        current_lines = frame_result.lines
        start_ms = frame_result.timestamp_ms
        last_seen_ms = frame_result.timestamp_ms
        confidences = [confidence]

    close_current()
    return segments


def merge_repeated_subtitles(
    frame_results: list[FrameOcrResult],
    frame_interval_ms: int,
    similarity_threshold: float = 0.92,
    max_gap_ms: int | None = None,
) -> list[SubtitleSegment]:
    return merge_repeated_text_snapshots(
        frame_results=frame_results,
        frame_interval_ms=frame_interval_ms,
        similarity_threshold=similarity_threshold,
        max_gap_ms=max_gap_ms,
    )
