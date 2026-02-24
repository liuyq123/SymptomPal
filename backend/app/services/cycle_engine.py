"""
Cycle computation engine: derives cycles from period start dates,
tags symptom logs with cycle days, and detects symptom-cycle correlations
using phase-aware alignment (Reverse Cycle Day for luteal-phase patterns).

Key design: cycle days are COMPUTED, not stored. CycleDayTag is derived
on-the-fly from cycle boundaries, so retroactive cycle entry immediately
enriches all historical symptom logs.
"""

from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..models import (
    CycleDayLog, CycleInfo, CycleDayTag, CycleSymptomCorrelation,
    CyclePatternReport, LogEntry,
)


def _compute_phase(cycle_day: int, cycle_length: int = 28) -> str:
    """Determine cycle phase from day number.

    The luteal phase is relatively fixed at ~14 days,
    so ovulation ≈ cycle_length - 14.
    """
    if cycle_day <= 5:
        return "menstrual"
    ovulation_day = max(6, cycle_length - 14)
    if cycle_day < ovulation_day - 1:
        return "follicular"
    if cycle_day <= ovulation_day + 1:
        return "ovulatory"
    return "luteal"


def compute_cycles(cycle_logs: List[CycleDayLog]) -> List[CycleInfo]:
    """Derive cycle boundaries from period day logs.

    Algorithm:
    1. Sort all logged days chronologically
    2. Find period start dates: a period day preceded by a gap of 5+ days
    3. Each cycle runs from one period start to the day before the next
    """
    if not cycle_logs:
        return []

    sorted_logs = sorted(cycle_logs, key=lambda x: x.date)
    period_days = [log for log in sorted_logs if log.is_period_day]

    if not period_days:
        return []

    # Find period start dates (gaps of 5+ non-period days indicate new period)
    period_starts: List[str] = []
    last_period_date: Optional[date] = None

    for log in period_days:
        log_date = date.fromisoformat(log.date)
        if last_period_date is None:
            period_starts.append(log.date)
        else:
            gap = (log_date - last_period_date).days
            if gap > 5:  # New period if gap > 5 days
                period_starts.append(log.date)
        last_period_date = log_date

    # Build CycleInfo objects
    cycles = []
    period_day_dates = {log.date for log in period_days}

    for i, start_str in enumerate(period_starts):
        start = date.fromisoformat(start_str)
        if i + 1 < len(period_starts):
            next_start = date.fromisoformat(period_starts[i + 1])
            end = next_start - timedelta(days=1)
            length = (next_start - start).days
        else:
            end = None  # Current/open cycle
            length = None

        # Count period days in this cycle
        period_length = 0
        check_date = start
        while check_date.isoformat() in period_day_dates:
            period_length += 1
            check_date += timedelta(days=1)
            if period_length > 14:  # Safety cap
                break

        cycles.append(CycleInfo(
            cycle_number=i + 1,
            start_date=start_str,
            end_date=end.isoformat() if end else None,
            length_days=length,
            period_length_days=max(period_length, 1),
        ))

    return cycles


def tag_log_with_cycle_day(
    log_recorded_at: datetime,
    cycles: List[CycleInfo],
) -> Optional[CycleDayTag]:
    """Compute the cycle day for a given symptom log timestamp.

    Returns None if the log falls outside any known cycle window.
    For the current (open) cycle, allows up to 45 days.
    """
    if not cycles:
        return None

    if isinstance(log_recorded_at, datetime):
        log_date = log_recorded_at.date()
    else:
        log_date = log_recorded_at

    for cycle in reversed(cycles):  # Check most recent first
        cycle_start = date.fromisoformat(cycle.start_date)

        if cycle.end_date:
            cycle_end = date.fromisoformat(cycle.end_date)
        else:
            # Open cycle: allow up to 45 days
            cycle_end = cycle_start + timedelta(days=45)

        if cycle_start <= log_date <= cycle_end:
            cycle_day = (log_date - cycle_start).days + 1  # 1-indexed
            cycle_length = cycle.length_days or 28
            phase = _compute_phase(cycle_day, cycle_length)

            return CycleDayTag(
                cycle_day=cycle_day,
                cycle_phase=phase,
                cycle_number=cycle.cycle_number,
                cycle_start_date=cycle.start_date,
            )

    return None


def _cluster_days(days: List[int], tolerance: int = 2) -> List[List[int]]:
    """Cluster cycle days that are within tolerance of each other."""
    if not days:
        return []
    clusters: List[List[int]] = []
    current_cluster = [days[0]]
    for day in days[1:]:
        if day - current_cluster[-1] <= tolerance:
            current_cluster.append(day)
        else:
            clusters.append(current_cluster)
            current_cluster = [day]
    clusters.append(current_cluster)
    return clusters


def detect_correlations(
    logs: List[LogEntry],
    cycles: List[CycleInfo],
    min_cycles: int = 2,
) -> List[CycleSymptomCorrelation]:
    """Detect symptom-cycle correlations across multiple cycles.

    Algorithm:
    1. Tag every symptom log with its cycle day
    2. Compute phase-aware alignment days:
       - Menstrual/follicular/ovulatory: forward cycle day (anchored to period start)
       - Luteal: Reverse Cycle Day (anchored to next period start)
       The luteal phase is ~14 days regardless of cycle length, so RCD alignment
       reveals phase-locked patterns that forward counting misses in variable cycles.
    3. Cluster by alignment day to detect recurring patterns across cycles
    """
    if len(cycles) < min_cycles:
        return []

    completed_cycles = [c for c in cycles if c.length_days is not None]
    if len(completed_cycles) < min_cycles:
        completed_cycles = cycles

    total_cycles = len(completed_cycles)

    # Cycle length map for Reverse Cycle Day computation
    cycle_length_map = {c.cycle_number: c.length_days for c in cycles if c.length_days}
    avg_cycle_length = (
        sum(cycle_length_map.values()) / len(cycle_length_map)
        if cycle_length_map else 28
    )

    # Step 1: Tag all logs and compute phase-aware alignment days
    # symptom_name -> {cycle_number -> [(forward_cd, alignment_day)]}
    symptom_cycle_data: Dict[str, Dict[int, List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for log in logs:
        tag = tag_log_with_cycle_day(log.recorded_at, cycles)
        if tag is None:
            continue
        cl = cycle_length_map.get(tag.cycle_number, int(avg_cycle_length))
        phase = _compute_phase(tag.cycle_day, cl)

        # Phase-aware alignment: forward for early phases, reverse for luteal
        if phase == "luteal":
            alignment_day = -(cl - tag.cycle_day)  # Negative = days before next period
        else:
            alignment_day = tag.cycle_day  # Forward from period start

        for symptom in log.extracted.symptoms:
            name = symptom.symptom.lower()
            symptom_cycle_data[name][tag.cycle_number].append(
                (tag.cycle_day, alignment_day)
            )

    # Step 2: Cluster by alignment day to find phase-locked patterns
    correlations = []

    for symptom_name, cycles_map in symptom_cycle_data.items():
        if len(cycles_map) < min_cycles:
            continue

        all_entries: List[Tuple[int, int, int]] = []  # (cycle_num, fwd_cd, align_day)
        for cycle_num, pairs in cycles_map.items():
            for fwd, align in pairs:
                all_entries.append((cycle_num, fwd, align))

        align_values = sorted(set(a for _, _, a in all_entries))
        clusters = _cluster_days(align_values, tolerance=2)

        for cluster_aligns in clusters:
            cycles_with_symptom = set()
            for cycle_num, fwd, align in all_entries:
                if any(abs(align - ca) <= 2 for ca in cluster_aligns):
                    cycles_with_symptom.add(cycle_num)

            occurrences = len(cycles_with_symptom)
            if occurrences < min_cycles:
                continue

            ratio = occurrences / total_cycles
            if ratio >= 0.75:
                confidence = "strong"
            elif ratio >= 0.50:
                confidence = "moderate"
            else:
                confidence = "weak"

            # Forward cycle days for backward compatibility
            cluster_fwd_days = sorted(set(
                fwd for _, fwd, align in all_entries
                if any(abs(align - ca) <= 2 for ca in cluster_aligns)
            ))

            # Generate description based on alignment type
            if all(a <= 0 for a in cluster_aligns):
                # Luteal phase: describe as days before menstruation
                onset_days = abs(min(cluster_aligns))
                primary_phase = "luteal"
                description = (
                    f"{symptom_name.title()} onset {onset_days} days "
                    f"before menstruation "
                    f"in {occurrences} of {total_cycles} cycles "
                    f"({primary_phase} phase)"
                )
            elif all(a > 0 for a in cluster_aligns):
                # Early cycle: describe as forward cycle days
                primary_phase = _compute_phase(
                    int(sum(cluster_aligns) / len(cluster_aligns)),
                    int(avg_cycle_length),
                )
                day_range = (
                    f"Days {min(cluster_aligns)}-{max(cluster_aligns)}"
                    if len(cluster_aligns) > 1
                    else f"Day {cluster_aligns[0]}"
                )
                description = (
                    f"{symptom_name.title()} on {day_range} "
                    f"in {occurrences} of {total_cycles} cycles "
                    f"({primary_phase} phase)"
                )
            else:
                # Mixed: symptom spans late luteal into menstruation
                before = abs(min(cluster_aligns))
                after = max(cluster_aligns)
                primary_phase = "luteal"
                description = (
                    f"{symptom_name.title()} from {before} days before "
                    f"through day {after} of menstruation "
                    f"in {occurrences} of {total_cycles} cycles "
                    f"(luteal-menstrual)"
                )

            correlations.append(CycleSymptomCorrelation(
                symptom=symptom_name,
                cycle_days=cluster_fwd_days,
                cycle_phase=primary_phase,
                occurrences=occurrences,
                total_cycles=total_cycles,
                confidence=confidence,
                description=description,
            ))

    # Sort by confidence strength then occurrences
    confidence_order = {"strong": 0, "moderate": 1, "weak": 2}
    correlations.sort(key=lambda c: (confidence_order[c.confidence], -c.occurrences))

    return correlations
