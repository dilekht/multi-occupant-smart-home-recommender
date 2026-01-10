"""
Conflict Resolution Module for Multi-Resident Smart Home Recommendations
=========================================================================

This module implements conflict detection, classification, and resolution
strategies for multi-occupant smart home scenarios.

Integrates with:
- FP-Growth discovered patterns
- Extended GLM predictions
- ARAS dataset preprocessing

Resolution Strategies:
1. Priority-Based: Activity criticality determines winner
2. Compromise: Find middle-ground settings
3. Temporal: Schedule-based conflict avoidance
4. Spatial: Location-based separation
5. Device-Specific: Smart device adjustments

Author: Research Extension Project
Date: January 2026
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import json
from datetime import datetime, timedelta
import warnings


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ConflictType(Enum):
    """Types of conflicts between residents."""
    NOISE = "noise"  # Sound-related conflicts (TV, Music vs Sleep/Study)
    DISTRACTION = "distraction"  # Visual/attention conflicts (TV vs Study/Read)
    RESOURCE = "resource"  # Shared resource conflicts (Bathroom, Kitchen)
    TEMPERATURE = "temperature"  # Climate preference conflicts
    LIGHTING = "lighting"  # Light level preference conflicts
    NONE = "none"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResolutionStrategy(Enum):
    """Available resolution strategies."""
    PRIORITY = "priority"  # Higher priority activity wins
    COMPROMISE = "compromise"  # Find middle ground
    TEMPORAL = "temporal"  # Time-based scheduling
    SPATIAL = "spatial"  # Location separation
    DEVICE = "device"  # Smart device adjustment
    NEGOTIATE = "negotiate"  # User negotiation required


class DeviceType(Enum):
    """Smart home device types."""
    TV = "tv"
    SPEAKER = "speaker"
    LIGHT = "light"
    THERMOSTAT = "thermostat"
    BLINDS = "blinds"
    DOOR = "door"
    APPLIANCE = "appliance"


# Activity definitions with priorities and characteristics
ACTIVITY_PROFILES = {
    1: {"name": "Other", "priority": 1, "noise_tolerance": 5, "light_need": 3, "category": "other"},
    2: {"name": "Going Out", "priority": 2, "noise_tolerance": 5, "light_need": 3, "category": "away"},
    3: {"name": "Preparing Breakfast", "priority": 4, "noise_tolerance": 4, "light_need": 4, "category": "meal_prep"},
    4: {"name": "Having Breakfast", "priority": 3, "noise_tolerance": 4, "light_need": 4, "category": "meal"},
    5: {"name": "Preparing Lunch", "priority": 4, "noise_tolerance": 4, "light_need": 4, "category": "meal_prep"},
    6: {"name": "Having Lunch", "priority": 3, "noise_tolerance": 4, "light_need": 4, "category": "meal"},
    7: {"name": "Preparing Dinner", "priority": 4, "noise_tolerance": 4, "light_need": 4, "category": "meal_prep"},
    8: {"name": "Having Dinner", "priority": 3, "noise_tolerance": 4, "light_need": 4, "category": "meal"},
    9: {"name": "Washing Dishes", "priority": 2, "noise_tolerance": 5, "light_need": 4, "category": "household"},
    10: {"name": "Having Snack", "priority": 2, "noise_tolerance": 5, "light_need": 3, "category": "meal"},
    11: {"name": "Sleeping", "priority": 5, "noise_tolerance": 1, "light_need": 1, "category": "rest"},
    12: {"name": "Watching TV", "priority": 2, "noise_tolerance": 5, "light_need": 2, "category": "entertainment"},
    13: {"name": "Studying", "priority": 4, "noise_tolerance": 2, "light_need": 5, "category": "work"},
    14: {"name": "Having Shower", "priority": 4, "noise_tolerance": 5, "light_need": 4, "category": "hygiene"},
    15: {"name": "Toileting", "priority": 5, "noise_tolerance": 5, "light_need": 3, "category": "hygiene"},
    16: {"name": "Napping", "priority": 4, "noise_tolerance": 1, "light_need": 1, "category": "rest"},
    17: {"name": "Using Internet", "priority": 2, "noise_tolerance": 3, "light_need": 3, "category": "entertainment"},
    18: {"name": "Reading Book", "priority": 3, "noise_tolerance": 2, "light_need": 5, "category": "entertainment"},
    19: {"name": "Laundry", "priority": 2, "noise_tolerance": 5, "light_need": 3, "category": "household"},
    20: {"name": "Shaving", "priority": 3, "noise_tolerance": 4, "light_need": 5, "category": "hygiene"},
    21: {"name": "Brushing Teeth", "priority": 3, "noise_tolerance": 5, "light_need": 4, "category": "hygiene"},
    22: {"name": "Talking on Phone", "priority": 4, "noise_tolerance": 2, "light_need": 3, "category": "social"},
    23: {"name": "Listening to Music", "priority": 2, "noise_tolerance": 5, "light_need": 2, "category": "entertainment"},
    24: {"name": "Cleaning", "priority": 2, "noise_tolerance": 5, "light_need": 4, "category": "household"},
    25: {"name": "Having Conversation", "priority": 3, "noise_tolerance": 3, "light_need": 3, "category": "social"},
    26: {"name": "Having Guest", "priority": 3, "noise_tolerance": 4, "light_need": 4, "category": "social"},
    27: {"name": "Changing Clothes", "priority": 3, "noise_tolerance": 5, "light_need": 4, "category": "hygiene"},
}

# Conflict definitions: (activity1, activity2) -> conflict info
CONFLICT_MATRIX = {
    (11, 12): {"type": ConflictType.NOISE, "severity": ConflictSeverity.HIGH, "description": "Sleeping vs TV"},
    (11, 23): {"type": ConflictType.NOISE, "severity": ConflictSeverity.HIGH, "description": "Sleeping vs Music"},
    (11, 22): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Sleeping vs Phone"},
    (16, 12): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Napping vs TV"},
    (16, 23): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Napping vs Music"},
    (13, 12): {"type": ConflictType.DISTRACTION, "severity": ConflictSeverity.MEDIUM, "description": "Studying vs TV"},
    (13, 23): {"type": ConflictType.DISTRACTION, "severity": ConflictSeverity.LOW, "description": "Studying vs Music"},
    (13, 22): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Studying vs Phone"},
    (18, 12): {"type": ConflictType.DISTRACTION, "severity": ConflictSeverity.MEDIUM, "description": "Reading vs TV"},
    (18, 23): {"type": ConflictType.DISTRACTION, "severity": ConflictSeverity.LOW, "description": "Reading vs Music"},
    (22, 12): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Phone vs TV"},
    (22, 23): {"type": ConflictType.NOISE, "severity": ConflictSeverity.MEDIUM, "description": "Phone vs Music"},
    (14, 15): {"type": ConflictType.RESOURCE, "severity": ConflictSeverity.HIGH, "description": "Shower vs Toilet"},
    (14, 20): {"type": ConflictType.RESOURCE, "severity": ConflictSeverity.MEDIUM, "description": "Shower vs Shaving"},
    (14, 21): {"type": ConflictType.RESOURCE, "severity": ConflictSeverity.LOW, "description": "Shower vs Brushing"},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ResidentState:
    """Current state of a resident."""
    resident_id: int
    activity_id: int
    activity_name: str
    location: str
    priority: int
    noise_tolerance: int
    light_need: int
    start_time: datetime = None
    predicted_duration: int = 0  # seconds
    
    @classmethod
    def from_activity(cls, resident_id: int, activity_id: int, location: str = "unknown"):
        profile = ACTIVITY_PROFILES.get(activity_id, ACTIVITY_PROFILES[1])
        return cls(
            resident_id=resident_id,
            activity_id=activity_id,
            activity_name=profile["name"],
            location=location,
            priority=profile["priority"],
            noise_tolerance=profile["noise_tolerance"],
            light_need=profile["light_need"]
        )


@dataclass
class Conflict:
    """Detected conflict between residents."""
    conflict_id: str
    timestamp: datetime
    resident1: ResidentState
    resident2: ResidentState
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    location: str
    resolved: bool = False
    resolution: Optional['Resolution'] = None


@dataclass
class DeviceRecommendation:
    """Recommendation for a smart device."""
    device_type: DeviceType
    device_id: str
    action: str
    value: Any
    reason: str
    priority: int = 1


@dataclass
class Resolution:
    """Resolution for a conflict."""
    strategy: ResolutionStrategy
    description: str
    recommendations: List[DeviceRecommendation]
    affected_residents: List[int]
    confidence: float
    alternative_resolutions: List['Resolution'] = field(default_factory=list)


@dataclass
class RecommendationContext:
    """Context for generating recommendations."""
    timestamp: datetime
    hour: int
    time_of_day: str
    is_weekend: bool
    residents: List[ResidentState]
    active_conflicts: List[Conflict]
    location_occupancy: Dict[str, List[int]]
    device_states: Dict[str, Any]


# =============================================================================
# CONFLICT DETECTOR
# =============================================================================

class ConflictDetector:
    """
    Detects conflicts between residents based on their activities.
    
    Uses predefined conflict matrix and learned patterns from FP-Growth.
    """
    
    def __init__(self):
        self.conflict_matrix = CONFLICT_MATRIX
        self.learned_patterns: List[Dict] = []
        self.conflict_history: List[Conflict] = []
        self._conflict_counter = 0
        
    def load_patterns(self, patterns: List[Dict]) -> None:
        """Load conflict patterns from FP-Growth analysis."""
        self.learned_patterns = [p for p in patterns if 'CONFLICT' in str(p)]
        print(f"Loaded {len(self.learned_patterns)} conflict patterns")
    
    def detect_conflict(self, resident1: ResidentState, 
                        resident2: ResidentState,
                        location: str = "shared") -> Optional[Conflict]:
        """
        Detect if there's a conflict between two residents.
        
        Args:
            resident1: State of first resident
            resident2: State of second resident
            location: Current location context
            
        Returns:
            Conflict object if conflict detected, None otherwise
        """
        a1, a2 = resident1.activity_id, resident2.activity_id
        
        # Check both orderings in conflict matrix
        conflict_info = None
        if (a1, a2) in self.conflict_matrix:
            conflict_info = self.conflict_matrix[(a1, a2)]
        elif (a2, a1) in self.conflict_matrix:
            conflict_info = self.conflict_matrix[(a2, a1)]
        
        if conflict_info:
            self._conflict_counter += 1
            conflict = Conflict(
                conflict_id=f"CONF_{self._conflict_counter:06d}",
                timestamp=datetime.now(),
                resident1=resident1,
                resident2=resident2,
                conflict_type=conflict_info["type"],
                severity=conflict_info["severity"],
                description=conflict_info["description"],
                location=location
            )
            self.conflict_history.append(conflict)
            return conflict
        
        # Check noise tolerance mismatch
        noise_conflict = self._check_noise_conflict(resident1, resident2)
        if noise_conflict:
            return noise_conflict
        
        return None
    
    def _check_noise_conflict(self, r1: ResidentState, r2: ResidentState) -> Optional[Conflict]:
        """Check for implicit noise-based conflicts."""
        # If one has low noise tolerance and other is doing noisy activity
        noisy_activities = {12, 23, 19, 24}  # TV, Music, Laundry, Cleaning
        quiet_needed = {11, 16, 13, 18, 22}  # Sleep, Nap, Study, Read, Phone
        
        conflict = None
        
        if r1.activity_id in quiet_needed and r2.activity_id in noisy_activities:
            if r1.noise_tolerance <= 2:
                self._conflict_counter += 1
                conflict = Conflict(
                    conflict_id=f"CONF_{self._conflict_counter:06d}",
                    timestamp=datetime.now(),
                    resident1=r1,
                    resident2=r2,
                    conflict_type=ConflictType.NOISE,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"{r1.activity_name} disturbed by {r2.activity_name}",
                    location="shared"
                )
        elif r2.activity_id in quiet_needed and r1.activity_id in noisy_activities:
            if r2.noise_tolerance <= 2:
                self._conflict_counter += 1
                conflict = Conflict(
                    conflict_id=f"CONF_{self._conflict_counter:06d}",
                    timestamp=datetime.now(),
                    resident1=r1,
                    resident2=r2,
                    conflict_type=ConflictType.NOISE,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"{r2.activity_name} disturbed by {r1.activity_name}",
                    location="shared"
                )
        
        if conflict:
            self.conflict_history.append(conflict)
        
        return conflict
    
    def detect_from_predictions(self, activity_r1: int, activity_r2: int,
                                 location: str = "shared") -> Optional[Conflict]:
        """Detect conflict from GLM predictions."""
        r1 = ResidentState.from_activity(1, activity_r1, location)
        r2 = ResidentState.from_activity(2, activity_r2, location)
        return self.detect_conflict(r1, r2, location)
    
    def get_conflict_stats(self) -> Dict:
        """Get statistics about detected conflicts."""
        if not self.conflict_history:
            return {"total": 0}
        
        stats = {
            "total": len(self.conflict_history),
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "resolved": sum(1 for c in self.conflict_history if c.resolved),
            "unresolved": sum(1 for c in self.conflict_history if not c.resolved)
        }
        
        for conflict in self.conflict_history:
            stats["by_type"][conflict.conflict_type.value] += 1
            stats["by_severity"][conflict.severity.name] += 1
        
        stats["by_type"] = dict(stats["by_type"])
        stats["by_severity"] = dict(stats["by_severity"])
        
        return stats


# =============================================================================
# RESOLUTION STRATEGIES
# =============================================================================

class ResolutionStrategyBase:
    """Base class for resolution strategies."""
    
    def __init__(self):
        self.name = "base"
        
    def can_resolve(self, conflict: Conflict) -> bool:
        """Check if this strategy can resolve the conflict."""
        return True
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        """Generate resolution for the conflict."""
        raise NotImplementedError


class PriorityBasedResolution(ResolutionStrategyBase):
    """
    Resolution based on activity priority.
    Higher priority activity takes precedence.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "priority"
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        r1, r2 = conflict.resident1, conflict.resident2
        
        # Determine winner based on priority
        if r1.priority >= r2.priority:
            winner, loser = r1, r2
        else:
            winner, loser = r2, r1
        
        recommendations = []
        
        if conflict.conflict_type == ConflictType.NOISE:
            # Recommend noise reduction for loser's activity
            if loser.activity_id == 12:  # TV
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.TV,
                    device_id="living_room_tv",
                    action="set_volume",
                    value=15,  # Lower volume
                    reason=f"Reduce volume for {winner.activity_name}",
                    priority=1
                ))
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.TV,
                    device_id="living_room_tv",
                    action="enable_subtitles",
                    value=True,
                    reason="Enable subtitles to compensate for lower volume",
                    priority=2
                ))
            elif loser.activity_id == 23:  # Music
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.SPEAKER,
                    device_id="smart_speaker",
                    action="set_volume",
                    value=20,
                    reason=f"Reduce volume for {winner.activity_name}",
                    priority=1
                ))
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.SPEAKER,
                    device_id="smart_speaker",
                    action="suggest_headphones",
                    value=True,
                    reason="Suggest using headphones",
                    priority=2
                ))
        
        elif conflict.conflict_type == ConflictType.DISTRACTION:
            if loser.activity_id == 12:  # TV
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.TV,
                    device_id="living_room_tv",
                    action="reduce_brightness",
                    value=50,
                    reason=f"Reduce visual distraction for {winner.activity_name}",
                    priority=1
                ))
        
        elif conflict.conflict_type == ConflictType.RESOURCE:
            # Resource conflicts need scheduling
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.APPLIANCE,
                device_id="notification_system",
                action="schedule_notification",
                value={"message": f"Bathroom available in ~10 minutes", "target": loser.resident_id},
                reason="Notify when resource becomes available",
                priority=1
            ))
        
        return Resolution(
            strategy=ResolutionStrategy.PRIORITY,
            description=f"Priority resolution: {winner.activity_name} (priority {winner.priority}) takes precedence over {loser.activity_name} (priority {loser.priority})",
            recommendations=recommendations,
            affected_residents=[winner.resident_id, loser.resident_id],
            confidence=0.85
        )


class CompromiseResolution(ResolutionStrategyBase):
    """
    Resolution through compromise.
    Both residents adjust their preferences.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "compromise"
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        r1, r2 = conflict.resident1, conflict.resident2
        recommendations = []
        
        if conflict.conflict_type == ConflictType.NOISE:
            # Calculate compromise noise level
            avg_tolerance = (r1.noise_tolerance + r2.noise_tolerance) / 2
            compromise_volume = int(avg_tolerance * 10)  # 0-50 volume range
            
            if r2.activity_id == 12:  # R2 watching TV
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.TV,
                    device_id="living_room_tv",
                    action="set_volume",
                    value=compromise_volume,
                    reason=f"Compromise volume level ({compromise_volume}%)",
                    priority=1
                ))
            elif r1.activity_id == 12:  # R1 watching TV
                recommendations.append(DeviceRecommendation(
                    device_type=DeviceType.TV,
                    device_id="living_room_tv",
                    action="set_volume",
                    value=compromise_volume,
                    reason=f"Compromise volume level ({compromise_volume}%)",
                    priority=1
                ))
            
            # Suggest noise-canceling for the quiet activity
            quiet_resident = r1 if r1.noise_tolerance < r2.noise_tolerance else r2
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.APPLIANCE,
                device_id="notification_system",
                action="suggest",
                value={"message": "Consider using noise-canceling headphones", 
                       "target": quiet_resident.resident_id},
                reason="Help quiet activity resident",
                priority=2
            ))
        
        elif conflict.conflict_type == ConflictType.LIGHTING:
            avg_light = (r1.light_need + r2.light_need) / 2
            compromise_brightness = int(avg_light * 20)  # 0-100 brightness
            
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.LIGHT,
                device_id="main_light",
                action="set_brightness",
                value=compromise_brightness,
                reason=f"Compromise brightness level ({compromise_brightness}%)",
                priority=1
            ))
            
            # Suggest task light for high-need resident
            if r1.light_need > r2.light_need:
                target = r1.resident_id
            else:
                target = r2.resident_id
            
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.LIGHT,
                device_id="task_light",
                action="turn_on",
                value=True,
                reason="Provide additional task lighting",
                priority=2
            ))
        
        return Resolution(
            strategy=ResolutionStrategy.COMPROMISE,
            description=f"Compromise: Both residents adjust - {r1.activity_name} and {r2.activity_name}",
            recommendations=recommendations,
            affected_residents=[r1.resident_id, r2.resident_id],
            confidence=0.75
        )


class TemporalResolution(ResolutionStrategyBase):
    """
    Resolution through time-based scheduling.
    Suggest postponing one activity.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "temporal"
    
    def can_resolve(self, conflict: Conflict) -> bool:
        # Can't use temporal for urgent activities
        urgent_activities = {15, 14}  # Toileting, Shower
        return (conflict.resident1.activity_id not in urgent_activities and
                conflict.resident2.activity_id not in urgent_activities)
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        r1, r2 = conflict.resident1, conflict.resident2
        
        # Lower priority activity should be postponed
        if r1.priority < r2.priority:
            postpone, keep = r1, r2
        else:
            postpone, keep = r2, r1
        
        # Estimate duration based on activity type
        duration_estimates = {
            11: 28800,  # Sleep: 8 hours
            16: 1800,   # Nap: 30 min
            12: 3600,   # TV: 1 hour
            13: 7200,   # Study: 2 hours
            18: 3600,   # Reading: 1 hour
        }
        
        wait_time = duration_estimates.get(keep.activity_id, 1800) // 60  # minutes
        
        recommendations = [
            DeviceRecommendation(
                device_type=DeviceType.APPLIANCE,
                device_id="notification_system",
                action="schedule_reminder",
                value={
                    "message": f"Good time to start {postpone.activity_name}",
                    "delay_minutes": wait_time,
                    "target": postpone.resident_id
                },
                reason=f"Schedule {postpone.activity_name} for later",
                priority=1
            ),
            DeviceRecommendation(
                device_type=DeviceType.APPLIANCE,
                device_id="notification_system",
                action="suggest_alternative",
                value={
                    "message": f"Consider an alternative activity while waiting",
                    "suggestions": self._get_alternative_activities(postpone.activity_id),
                    "target": postpone.resident_id
                },
                reason="Suggest alternative activities",
                priority=2
            )
        ]
        
        return Resolution(
            strategy=ResolutionStrategy.TEMPORAL,
            description=f"Temporal: Postpone {postpone.activity_name} by ~{wait_time} minutes until {keep.activity_name} completes",
            recommendations=recommendations,
            affected_residents=[postpone.resident_id],
            confidence=0.70
        )
    
    def _get_alternative_activities(self, activity_id: int) -> List[str]:
        """Get alternative activities that don't conflict."""
        alternatives = {
            12: ["Using Internet", "Reading Book", "Having Snack"],  # Instead of TV
            23: ["Reading Book", "Using Internet"],  # Instead of Music
            13: ["Using Internet", "Having Snack"],  # Instead of Study
        }
        return alternatives.get(activity_id, ["Relaxing", "Having Snack"])


class SpatialResolution(ResolutionStrategyBase):
    """
    Resolution through spatial separation.
    Suggest moving to different locations.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "spatial"
        
        # Activity to preferred location mapping
        self.activity_locations = {
            11: ["bedroom"],
            12: ["living_room", "bedroom"],
            13: ["study", "bedroom", "living_room"],
            18: ["bedroom", "living_room", "study"],
            23: ["living_room", "bedroom"],
        }
    
    def can_resolve(self, conflict: Conflict) -> bool:
        # Spatial works best for noise/distraction conflicts
        return conflict.conflict_type in [ConflictType.NOISE, ConflictType.DISTRACTION]
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        r1, r2 = conflict.resident1, conflict.resident2
        
        # Find alternative locations
        r1_locations = set(self.activity_locations.get(r1.activity_id, ["any"]))
        r2_locations = set(self.activity_locations.get(r2.activity_id, ["any"]))
        
        # Suggest different rooms
        recommendations = []
        
        # Lower priority should move
        if r1.priority < r2.priority:
            mover, stayer = r1, r2
        else:
            mover, stayer = r2, r1
        
        alternative_locations = self.activity_locations.get(mover.activity_id, [])
        # Remove current location
        alternative_locations = [loc for loc in alternative_locations if loc != conflict.location]
        
        if alternative_locations:
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.APPLIANCE,
                device_id="notification_system",
                action="suggest_location",
                value={
                    "message": f"Consider moving to {alternative_locations[0]} for {mover.activity_name}",
                    "locations": alternative_locations,
                    "target": mover.resident_id
                },
                reason=f"Spatial separation for {mover.activity_name}",
                priority=1
            ))
            
            # Prepare the alternative location
            recommendations.append(DeviceRecommendation(
                device_type=DeviceType.LIGHT,
                device_id=f"{alternative_locations[0]}_light",
                action="set_brightness",
                value=mover.light_need * 20,
                reason=f"Prepare lighting in {alternative_locations[0]}",
                priority=2
            ))
        
        return Resolution(
            strategy=ResolutionStrategy.SPATIAL,
            description=f"Spatial: {mover.activity_name} relocate to {alternative_locations[0] if alternative_locations else 'another room'}",
            recommendations=recommendations,
            affected_residents=[mover.resident_id],
            confidence=0.80
        )


class DeviceSpecificResolution(ResolutionStrategyBase):
    """
    Resolution through smart device adjustments.
    Fine-grained control of specific devices.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "device"
    
    def resolve(self, conflict: Conflict, context: RecommendationContext) -> Resolution:
        r1, r2 = conflict.resident1, conflict.resident2
        recommendations = []
        
        if conflict.conflict_type == ConflictType.NOISE:
            # Detailed noise management
            noisy_resident = r2 if r2.activity_id in {12, 23} else r1
            quiet_resident = r1 if r1.noise_tolerance < r2.noise_tolerance else r2
            
            if noisy_resident.activity_id == 12:  # TV
                recommendations.extend([
                    DeviceRecommendation(
                        device_type=DeviceType.TV,
                        device_id="living_room_tv",
                        action="set_volume",
                        value=20,
                        reason="Reduce TV volume",
                        priority=1
                    ),
                    DeviceRecommendation(
                        device_type=DeviceType.TV,
                        device_id="living_room_tv",
                        action="enable_night_mode",
                        value=True,
                        reason="Enable night/quiet mode",
                        priority=2
                    ),
                    DeviceRecommendation(
                        device_type=DeviceType.TV,
                        device_id="living_room_tv",
                        action="enable_subtitles",
                        value=True,
                        reason="Enable subtitles for lower volume viewing",
                        priority=3
                    ),
                    DeviceRecommendation(
                        device_type=DeviceType.SPEAKER,
                        device_id="bluetooth_headphones",
                        action="suggest_use",
                        value={"target": noisy_resident.resident_id},
                        reason="Suggest wireless headphones for TV audio",
                        priority=4
                    )
                ])
            
            elif noisy_resident.activity_id == 23:  # Music
                recommendations.extend([
                    DeviceRecommendation(
                        device_type=DeviceType.SPEAKER,
                        device_id="smart_speaker",
                        action="set_volume",
                        value=15,
                        reason="Reduce music volume",
                        priority=1
                    ),
                    DeviceRecommendation(
                        device_type=DeviceType.SPEAKER,
                        device_id="smart_speaker",
                        action="suggest_headphones",
                        value=True,
                        reason="Suggest using headphones",
                        priority=2
                    )
                ])
            
            # For quiet activity, enhance isolation
            if quiet_resident.activity_id in {11, 16}:  # Sleep/Nap
                recommendations.extend([
                    DeviceRecommendation(
                        device_type=DeviceType.BLINDS,
                        device_id="bedroom_blinds",
                        action="close",
                        value=100,
                        reason="Close blinds for sleep",
                        priority=2
                    ),
                    DeviceRecommendation(
                        device_type=DeviceType.LIGHT,
                        device_id="bedroom_light",
                        action="turn_off",
                        value=True,
                        reason="Turn off lights for sleep",
                        priority=2
                    )
                ])
        
        elif conflict.conflict_type == ConflictType.LIGHTING:
            high_light = r1 if r1.light_need > r2.light_need else r2
            low_light = r2 if r1.light_need > r2.light_need else r1
            
            recommendations.extend([
                DeviceRecommendation(
                    device_type=DeviceType.LIGHT,
                    device_id="main_light",
                    action="set_brightness",
                    value=low_light.light_need * 15,  # Lower for low-light person
                    reason="Set ambient lighting for both",
                    priority=1
                ),
                DeviceRecommendation(
                    device_type=DeviceType.LIGHT,
                    device_id="task_light",
                    action="turn_on_for",
                    value={"brightness": high_light.light_need * 20, "target": high_light.resident_id},
                    reason="Provide task lighting for high-need activity",
                    priority=2
                )
            ])
        
        return Resolution(
            strategy=ResolutionStrategy.DEVICE,
            description=f"Device-specific adjustments for {conflict.conflict_type.value} conflict",
            recommendations=recommendations,
            affected_residents=[r1.resident_id, r2.resident_id],
            confidence=0.90
        )


# =============================================================================
# CONFLICT RESOLVER (MAIN CLASS)
# =============================================================================

class ConflictResolver:
    """
    Main conflict resolution engine.
    
    Coordinates detection, strategy selection, and resolution generation.
    """
    
    def __init__(self):
        self.detector = ConflictDetector()
        
        # Initialize resolution strategies
        self.strategies = {
            ResolutionStrategy.PRIORITY: PriorityBasedResolution(),
            ResolutionStrategy.COMPROMISE: CompromiseResolution(),
            ResolutionStrategy.TEMPORAL: TemporalResolution(),
            ResolutionStrategy.SPATIAL: SpatialResolution(),
            ResolutionStrategy.DEVICE: DeviceSpecificResolution(),
        }
        
        # Strategy selection rules based on conflict type and severity
        self.strategy_preferences = {
            ConflictType.NOISE: {
                ConflictSeverity.HIGH: [ResolutionStrategy.DEVICE, ResolutionStrategy.SPATIAL, ResolutionStrategy.PRIORITY],
                ConflictSeverity.MEDIUM: [ResolutionStrategy.COMPROMISE, ResolutionStrategy.DEVICE],
                ConflictSeverity.LOW: [ResolutionStrategy.COMPROMISE],
            },
            ConflictType.DISTRACTION: {
                ConflictSeverity.HIGH: [ResolutionStrategy.SPATIAL, ResolutionStrategy.PRIORITY],
                ConflictSeverity.MEDIUM: [ResolutionStrategy.DEVICE, ResolutionStrategy.COMPROMISE],
                ConflictSeverity.LOW: [ResolutionStrategy.COMPROMISE],
            },
            ConflictType.RESOURCE: {
                ConflictSeverity.HIGH: [ResolutionStrategy.PRIORITY, ResolutionStrategy.TEMPORAL],
                ConflictSeverity.MEDIUM: [ResolutionStrategy.TEMPORAL],
                ConflictSeverity.LOW: [ResolutionStrategy.TEMPORAL, ResolutionStrategy.COMPROMISE],
            },
            ConflictType.TEMPERATURE: {
                ConflictSeverity.HIGH: [ResolutionStrategy.COMPROMISE, ResolutionStrategy.SPATIAL],
                ConflictSeverity.MEDIUM: [ResolutionStrategy.COMPROMISE],
                ConflictSeverity.LOW: [ResolutionStrategy.COMPROMISE],
            },
            ConflictType.LIGHTING: {
                ConflictSeverity.HIGH: [ResolutionStrategy.DEVICE, ResolutionStrategy.SPATIAL],
                ConflictSeverity.MEDIUM: [ResolutionStrategy.DEVICE, ResolutionStrategy.COMPROMISE],
                ConflictSeverity.LOW: [ResolutionStrategy.COMPROMISE],
            },
        }
        
        # Resolution history
        self.resolution_history: List[Tuple[Conflict, Resolution]] = []
    
    def detect_and_resolve(self, 
                           activity_r1: int, 
                           activity_r2: int,
                           context: Optional[RecommendationContext] = None,
                           location: str = "shared") -> Optional[Tuple[Conflict, Resolution]]:
        """
        Detect conflict and generate resolution in one step.
        
        Args:
            activity_r1: Activity ID for resident 1
            activity_r2: Activity ID for resident 2
            context: Recommendation context
            location: Current location
            
        Returns:
            Tuple of (Conflict, Resolution) if conflict detected, None otherwise
        """
        # Create resident states
        r1 = ResidentState.from_activity(1, activity_r1, location)
        r2 = ResidentState.from_activity(2, activity_r2, location)
        
        # Detect conflict
        conflict = self.detector.detect_conflict(r1, r2, location)
        
        if conflict is None:
            return None
        
        # Create context if not provided
        if context is None:
            context = self._create_default_context(r1, r2)
        
        # Generate resolution
        resolution = self.resolve(conflict, context)
        
        # Store in history
        self.resolution_history.append((conflict, resolution))
        
        return conflict, resolution
    
    def resolve(self, conflict: Conflict, 
                context: RecommendationContext) -> Resolution:
        """
        Generate resolution for a detected conflict.
        
        Args:
            conflict: Detected conflict
            context: Recommendation context
            
        Returns:
            Resolution with recommendations
        """
        # Get preferred strategies for this conflict type/severity
        preferences = self.strategy_preferences.get(
            conflict.conflict_type, 
            {ConflictSeverity.MEDIUM: [ResolutionStrategy.COMPROMISE]}
        ).get(conflict.severity, [ResolutionStrategy.COMPROMISE])
        
        # Try strategies in order of preference
        primary_resolution = None
        alternative_resolutions = []
        
        for strategy_type in preferences:
            strategy = self.strategies.get(strategy_type)
            if strategy and strategy.can_resolve(conflict):
                resolution = strategy.resolve(conflict, context)
                
                if primary_resolution is None:
                    primary_resolution = resolution
                else:
                    alternative_resolutions.append(resolution)
        
        # Fallback to priority-based if nothing else works
        if primary_resolution is None:
            primary_resolution = self.strategies[ResolutionStrategy.PRIORITY].resolve(conflict, context)
        
        # Add alternatives
        primary_resolution.alternative_resolutions = alternative_resolutions[:2]  # Keep top 2
        
        # Mark conflict as resolved
        conflict.resolved = True
        conflict.resolution = primary_resolution
        
        return primary_resolution
    
    def _create_default_context(self, r1: ResidentState, r2: ResidentState) -> RecommendationContext:
        """Create default context from resident states."""
        now = datetime.now()
        hour = now.hour
        
        if 0 <= hour < 6:
            tod = "Night"
        elif 6 <= hour < 12:
            tod = "Morning"
        elif 12 <= hour < 18:
            tod = "Afternoon"
        else:
            tod = "Evening"
        
        return RecommendationContext(
            timestamp=now,
            hour=hour,
            time_of_day=tod,
            is_weekend=now.weekday() >= 5,
            residents=[r1, r2],
            active_conflicts=[],
            location_occupancy={"shared": [1, 2]},
            device_states={}
        )
    
    def get_recommendations_summary(self, resolution: Resolution) -> str:
        """Generate human-readable summary of recommendations."""
        lines = [
            f"Resolution Strategy: {resolution.strategy.value.upper()}",
            f"Description: {resolution.description}",
            f"Confidence: {resolution.confidence:.0%}",
            "",
            "Recommendations:"
        ]
        
        for i, rec in enumerate(resolution.recommendations, 1):
            lines.append(f"  {i}. [{rec.device_type.value.upper()}] {rec.action}: {rec.value}")
            lines.append(f"     Reason: {rec.reason}")
        
        if resolution.alternative_resolutions:
            lines.append("")
            lines.append("Alternative Resolutions:")
            for alt in resolution.alternative_resolutions:
                lines.append(f"  - {alt.strategy.value}: {alt.description}")
        
        return "\n".join(lines)
    
    def batch_resolve(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Process batch predictions and add conflict resolutions.
        
        Args:
            predictions: DataFrame with Activity_R1, Activity_R2 columns
            
        Returns:
            DataFrame with conflict and resolution info added
        """
        results = []
        
        for idx, row in predictions.iterrows():
            activity_r1 = int(row.get('Activity_R1', row.get('activity_r1', 1)))
            activity_r2 = int(row.get('Activity_R2', row.get('activity_r2', 1)))
            
            result = self.detect_and_resolve(activity_r1, activity_r2)
            
            if result:
                conflict, resolution = result
                results.append({
                    'index': idx,
                    'activity_r1': activity_r1,
                    'activity_r2': activity_r2,
                    'has_conflict': True,
                    'conflict_type': conflict.conflict_type.value,
                    'conflict_severity': conflict.severity.name,
                    'resolution_strategy': resolution.strategy.value,
                    'resolution_confidence': resolution.confidence,
                    'num_recommendations': len(resolution.recommendations),
                    'primary_recommendation': resolution.recommendations[0].action if resolution.recommendations else None
                })
            else:
                results.append({
                    'index': idx,
                    'activity_r1': activity_r1,
                    'activity_r2': activity_r2,
                    'has_conflict': False,
                    'conflict_type': None,
                    'conflict_severity': None,
                    'resolution_strategy': None,
                    'resolution_confidence': None,
                    'num_recommendations': 0,
                    'primary_recommendation': None
                })
        
        return pd.DataFrame(results)
    
    def get_statistics(self) -> Dict:
        """Get resolution statistics."""
        stats = self.detector.get_conflict_stats()
        
        if self.resolution_history:
            strategy_counts = defaultdict(int)
            confidence_sum = 0
            
            for conflict, resolution in self.resolution_history:
                strategy_counts[resolution.strategy.value] += 1
                confidence_sum += resolution.confidence
            
            stats['resolution_stats'] = {
                'total_resolved': len(self.resolution_history),
                'by_strategy': dict(strategy_counts),
                'avg_confidence': confidence_sum / len(self.resolution_history)
            }
        
        return stats


# =============================================================================
# SMART HOME RECOMMENDATION ENGINE
# =============================================================================

class SmartHomeRecommendationEngine:
    """
    Complete recommendation engine integrating GLM predictions with conflict resolution.
    
    This is the main class to use for generating smart home recommendations.
    """
    
    def __init__(self, glm_model=None):
        """
        Initialize the recommendation engine.
        
        Args:
            glm_model: Trained MultiResidentGLM model (optional)
        """
        self.glm_model = glm_model
        self.conflict_resolver = ConflictResolver()
        self.recommendation_history: List[Dict] = []
        
    def set_model(self, glm_model) -> None:
        """Set the GLM model for predictions."""
        self.glm_model = glm_model
    
    def predict_and_recommend(self, features: pd.DataFrame) -> List[Dict]:
        """
        Generate predictions and recommendations for given features.
        
        Args:
            features: Feature DataFrame (single row or multiple)
            
        Returns:
            List of recommendation dictionaries
        """
        if self.glm_model is None:
            raise ValueError("No GLM model set. Call set_model() first.")
        
        # Get predictions
        predictions = self.glm_model.predict(features)
        
        results = []
        
        for i in range(len(features)):
            activity_r1 = predictions['Activity_R1'][i]
            activity_r2 = predictions['Activity_R2'][i]
            conflict_prob = predictions.get('Conflict_Proba', [0])[i]
            
            # Detect and resolve conflicts
            conflict_result = self.conflict_resolver.detect_and_resolve(
                activity_r1, activity_r2
            )
            
            result = {
                'prediction': {
                    'activity_r1': int(activity_r1),
                    'activity_r1_name': ACTIVITY_PROFILES.get(int(activity_r1), {}).get('name', 'Unknown'),
                    'activity_r2': int(activity_r2),
                    'activity_r2_name': ACTIVITY_PROFILES.get(int(activity_r2), {}).get('name', 'Unknown'),
                    'conflict_probability': float(conflict_prob)
                },
                'conflict': None,
                'resolution': None,
                'device_recommendations': []
            }
            
            if conflict_result:
                conflict, resolution = conflict_result
                result['conflict'] = {
                    'type': conflict.conflict_type.value,
                    'severity': conflict.severity.name,
                    'description': conflict.description
                }
                result['resolution'] = {
                    'strategy': resolution.strategy.value,
                    'description': resolution.description,
                    'confidence': resolution.confidence
                }
                result['device_recommendations'] = [
                    {
                        'device': rec.device_type.value,
                        'device_id': rec.device_id,
                        'action': rec.action,
                        'value': rec.value,
                        'reason': rec.reason
                    }
                    for rec in resolution.recommendations
                ]
            
            results.append(result)
            self.recommendation_history.append(result)
        
        return results
    
    def recommend_from_activities(self, activity_r1: int, activity_r2: int) -> Dict:
        """
        Generate recommendations directly from activity IDs.
        
        Args:
            activity_r1: Activity ID for resident 1
            activity_r2: Activity ID for resident 2
            
        Returns:
            Recommendation dictionary
        """
        conflict_result = self.conflict_resolver.detect_and_resolve(
            activity_r1, activity_r2
        )
        
        result = {
            'activity_r1': activity_r1,
            'activity_r1_name': ACTIVITY_PROFILES.get(activity_r1, {}).get('name', 'Unknown'),
            'activity_r2': activity_r2,
            'activity_r2_name': ACTIVITY_PROFILES.get(activity_r2, {}).get('name', 'Unknown'),
            'has_conflict': conflict_result is not None,
            'conflict': None,
            'resolution': None,
            'recommendations': []
        }
        
        if conflict_result:
            conflict, resolution = conflict_result
            result['conflict'] = {
                'type': conflict.conflict_type.value,
                'severity': conflict.severity.name,
                'description': conflict.description
            }
            result['resolution'] = {
                'strategy': resolution.strategy.value,
                'description': resolution.description,
                'confidence': resolution.confidence,
                'alternatives': [
                    {'strategy': alt.strategy.value, 'description': alt.description}
                    for alt in resolution.alternative_resolutions
                ]
            }
            result['recommendations'] = [
                {
                    'device': rec.device_type.value,
                    'device_id': rec.device_id,
                    'action': rec.action,
                    'value': rec.value,
                    'reason': rec.reason,
                    'priority': rec.priority
                }
                for rec in resolution.recommendations
            ]
        
        return result
    
    def get_summary(self) -> Dict:
        """Get summary statistics of recommendations."""
        stats = self.conflict_resolver.get_statistics()
        stats['total_recommendations'] = len(self.recommendation_history)
        
        if self.recommendation_history:
            conflicts_detected = sum(1 for r in self.recommendation_history if r.get('conflict'))
            stats['recommendations_with_conflicts'] = conflicts_detected
            stats['conflict_rate'] = conflicts_detected / len(self.recommendation_history)
        
        return stats
    
    def export_recommendations(self, filepath: str) -> None:
        """Export recommendation history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.recommendation_history, f, indent=2, default=str)
        print(f"Exported {len(self.recommendation_history)} recommendations to {filepath}")


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo():
    """Demonstrate the conflict resolution module."""
    print("=" * 70)
    print("CONFLICT RESOLUTION MODULE DEMO")
    print("=" * 70)
    
    # Initialize engine
    engine = SmartHomeRecommendationEngine()
    
    # Test scenarios
    scenarios = [
        (11, 12, "Resident 1 Sleeping, Resident 2 Watching TV"),
        (13, 23, "Resident 1 Studying, Resident 2 Listening to Music"),
        (11, 23, "Resident 1 Sleeping, Resident 2 Listening to Music"),
        (18, 12, "Resident 1 Reading, Resident 2 Watching TV"),
        (14, 15, "Resident 1 Showering, Resident 2 Using Toilet"),
        (13, 17, "Resident 1 Studying, Resident 2 Using Internet (no conflict)"),
        (11, 11, "Both Sleeping (no conflict)"),
    ]
    
    print("\n" + "-" * 70)
    print("SCENARIO TESTING")
    print("-" * 70)
    
    for activity_r1, activity_r2, description in scenarios:
        print(f"\n📋 Scenario: {description}")
        print("-" * 50)
        
        result = engine.recommend_from_activities(activity_r1, activity_r2)
        
        print(f"   R1: {result['activity_r1_name']} (ID: {activity_r1})")
        print(f"   R2: {result['activity_r2_name']} (ID: {activity_r2})")
        
        if result['has_conflict']:
            print(f"\n   ⚠️  CONFLICT DETECTED")
            print(f"   Type: {result['conflict']['type']}")
            print(f"   Severity: {result['conflict']['severity']}")
            print(f"   Description: {result['conflict']['description']}")
            
            print(f"\n   ✅ RESOLUTION")
            print(f"   Strategy: {result['resolution']['strategy']}")
            print(f"   Confidence: {result['resolution']['confidence']:.0%}")
            print(f"   {result['resolution']['description']}")
            
            print(f"\n   📱 DEVICE RECOMMENDATIONS:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"   {i}. [{rec['device'].upper()}] {rec['action']}: {rec['value']}")
                print(f"      Reason: {rec['reason']}")
            
            if result['resolution'].get('alternatives'):
                print(f"\n   🔄 ALTERNATIVES:")
                for alt in result['resolution']['alternatives']:
                    print(f"   - {alt['strategy']}: {alt['description']}")
        else:
            print(f"\n   ✓ No conflict detected")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    stats = engine.get_summary()
    print(f"\nTotal scenarios tested: {stats['total_recommendations']}")
    print(f"Conflicts detected: {stats.get('recommendations_with_conflicts', 0)}")
    print(f"Conflict rate: {stats.get('conflict_rate', 0):.1%}")
    
    if 'resolution_stats' in stats:
        print(f"\nResolution strategies used:")
        for strategy, count in stats['resolution_stats']['by_strategy'].items():
            print(f"  - {strategy}: {count}")
    
    return engine


if __name__ == "__main__":
    engine = demo()
