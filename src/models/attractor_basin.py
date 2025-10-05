#!/usr/bin/env python3
"""
AttractorBasin Model: Context Engineering attractor basin representation
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class BasinType(str, Enum):
    """Types of attractor basins"""
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    SEMANTIC = "semantic"
    PATTERN = "pattern"
    MEMORY = "memory"
    ATTENTION = "attention"
    CONSCIOUSNESS = "consciousness"
    HYBRID = "hybrid"


class BasinState(str, Enum):
    """Attractor basin states"""
    DORMANT = "dormant"
    ACTIVATING = "activating"
    ACTIVE = "active"
    SATURATED = "saturated"
    DECAYING = "decaying"
    INHIBITED = "inhibited"


class ConnectionType(str, Enum):
    """Types of basin connections"""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    BIDIRECTIONAL = "bidirectional"


class BasinConnection(BaseModel):
    """Connection between attractor basins"""
    basin_id: str = Field(..., description="Connected basin ID")
    connection_strength: float = Field(..., ge=0.0, le=1.0, description="Connection strength")
    connection_type: ConnectionType = Field(..., description="Type of connection")
    
    # Connection dynamics
    latency_ms: float = Field(default=0.0, ge=0.0, description="Connection latency in milliseconds")
    bandwidth: float = Field(default=1.0, ge=0.0, le=1.0, description="Information bandwidth")
    plasticity: float = Field(default=0.1, ge=0.0, le=1.0, description="Connection plasticity")
    
    # Learning and adaptation
    learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Connection learning rate")
    last_update: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class NeuralFieldInfluence(BaseModel):
    """Neural field influence metrics"""
    field_contribution: float = Field(..., ge=0.0, le=1.0, description="Contribution to neural field")
    spatial_extent: float = Field(..., ge=0.0, description="Spatial extent of influence")
    temporal_persistence: float = Field(..., ge=0.0, description="Temporal persistence of influence")
    
    # Field dynamics
    gradient_strength: float = Field(default=0.0, ge=0.0, description="Field gradient strength")
    oscillation_frequency: float = Field(default=0.0, ge=0.0, description="Oscillation frequency")
    phase_coupling: float = Field(default=0.0, ge=-1.0, le=1.0, description="Phase coupling with global field")
    
    # Influence patterns
    influence_pattern: Dict[str, float] = Field(default_factory=dict, description="Spatial influence pattern")
    temporal_evolution: List[float] = Field(default_factory=list, description="Temporal evolution of influence")


class AttractorBasin(BaseModel):
    """
    AttractorBasin: Context Engineering attractor basin with neural field integration
    
    Represents stable states in the consciousness state space that attract
    and stabilize cognitive processes, integrated with ThoughtSeed hierarchy.
    """
    
    # Core identification
    basin_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique basin identifier")
    basin_name: str = Field(..., min_length=1, max_length=200, description="Human-readable basin name")
    basin_type: BasinType = Field(..., description="Type of attractor basin")
    description: Optional[str] = Field(None, description="Basin description")
    
    # Basin dynamics
    stability: float = Field(..., ge=0.0, le=1.0, description="Basin stability measure")
    depth: float = Field(..., gt=0.0, description="Basin depth (energy well depth)")
    width: float = Field(default=1.0, gt=0.0, description="Basin width (attraction radius)")
    
    # Activation and state
    current_state: BasinState = Field(default=BasinState.DORMANT, description="Current basin state")
    activation_threshold: float = Field(..., ge=0.0, le=1.0, description="Activation threshold")
    current_activation: float = Field(default=0.0, ge=0.0, le=1.0, description="Current activation level")
    saturation_level: float = Field(default=1.0, ge=0.0, le=1.0, description="Saturation activation level")
    
    # Basin connections
    connected_basins: List[BasinConnection] = Field(default_factory=list, description="Connected basins")
    connection_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Connection matrix")
    
    # Neural field integration
    neural_field_influence: NeuralFieldInfluence = Field(..., description="Neural field influence metrics")
    field_coordinates: List[float] = Field(default_factory=list, description="Coordinates in neural field space")
    
    # Consciousness integration
    consciousness_contribution: float = Field(default=0.0, ge=0.0, le=1.0, description="Contribution to consciousness")
    consciousness_markers: List[str] = Field(default_factory=list, description="Consciousness markers")
    emergence_indicators: Dict[str, float] = Field(default_factory=dict, description="Emergence indicator strengths")
    
    # ThoughtSeed integration
    thoughtseed_associations: List[str] = Field(default_factory=list, description="Associated ThoughtSeed trace IDs")
    layer_influences: Dict[str, float] = Field(
        default_factory=lambda: {
            "sensory": 0.0, "perceptual": 0.0, "conceptual": 0.0, 
            "abstract": 0.0, "metacognitive": 0.0
        },
        description="Influence on ThoughtSeed layers"
    )
    
    # Pattern and content associations
    associated_patterns: List[str] = Field(default_factory=list, description="Associated pattern IDs")
    pattern_strengths: Dict[str, float] = Field(default_factory=dict, description="Pattern association strengths")
    content_signatures: List[str] = Field(default_factory=list, description="Content signature hashes")
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Basin creation timestamp")
    last_activated: Optional[datetime] = Field(None, description="Last activation timestamp")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    activation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Activation history")

    # CLAUSE Phase 1: Basin Strengthening (Spec 034)
    strength: float = Field(default=1.0, ge=1.0, le=2.0, description="Basin frequency strength (1.0-2.0)")
    activation_count: int = Field(default=0, ge=0, description="Number of times basin activated")
    co_occurring_concepts: Dict[str, int] = Field(default_factory=dict, description="Co-occurrence counts with other concepts")
    
    # Learning and adaptation
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Basin learning rate")
    adaptation_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Adaptation strength")
    memory_trace: Dict[str, Any] = Field(default_factory=dict, description="Memory trace information")
    
    # Energy and dynamics
    energy_level: float = Field(default=0.0, description="Current energy level")
    dissipation_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Energy dissipation rate")
    convergence_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Convergence rate to attractor")
    
    # Semantic representation
    semantic_embedding: Optional[List[float]] = Field(None, description="Semantic embedding vector")
    conceptual_features: Dict[str, float] = Field(default_factory=dict, description="Conceptual feature weights")
    
    # Quality and validation
    validation_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Basin validation score")
    coherence_measure: float = Field(default=0.0, ge=0.0, le=1.0, description="Internal coherence measure")
    
    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "auto_decay": True,
            "learning_enabled": True,
            "consciousness_tracking": True,
            "field_influence_enabled": True
        },
        description="Basin configuration"
    )
    
    @validator('layer_influences')
    def validate_layer_influences(cls, v):
        """Validate ThoughtSeed layer influences"""
        required_layers = {"sensory", "perceptual", "conceptual", "abstract", "metacognitive"}
        if set(v.keys()) != required_layers:
            raise ValueError(f"Must have influences for all layers: {required_layers}")
        
        for layer, influence in v.items():
            if not isinstance(influence, (int, float)) or not (0.0 <= influence <= 1.0):
                raise ValueError(f"Layer influence must be between 0.0 and 1.0")
        return v
    
    @validator('semantic_embedding')
    def validate_semantic_embedding(cls, v):
        """Validate semantic embedding"""
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Semantic embedding must be list of numbers")
            if len(v) == 0:
                raise ValueError("Semantic embedding cannot be empty")
        return v
    
    def activate(self, activation_strength: float = 1.0) -> bool:
        """Activate the attractor basin"""
        if self.current_state == BasinState.INHIBITED:
            return False
        
        # Update activation level
        self.current_activation = min(
            self.current_activation + activation_strength,
            self.saturation_level
        )
        
        # Update state based on activation level
        if self.current_activation >= self.activation_threshold:
            if self.current_activation >= self.saturation_level:
                self.current_state = BasinState.SATURATED
            else:
                self.current_state = BasinState.ACTIVE
            
            self.last_activated = datetime.utcnow()
            
            # Record activation in history
            activation_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "activation_strength": activation_strength,
                "resulting_activation": self.current_activation,
                "state": self.current_state.value
            }
            self.activation_history.append(activation_event)
            
            return True
        else:
            self.current_state = BasinState.ACTIVATING
            return False
    
    def deactivate(self, decay_rate: Optional[float] = None) -> None:
        """Deactivate basin with optional custom decay rate"""
        if decay_rate is None:
            decay_rate = self.dissipation_rate
        
        self.current_activation = max(0.0, self.current_activation - decay_rate)
        
        if self.current_activation < self.activation_threshold * 0.1:
            self.current_state = BasinState.DORMANT
            self.current_activation = 0.0
        elif self.current_activation < self.activation_threshold:
            self.current_state = BasinState.DECAYING
    
    def add_connection(self, target_basin_id: str, connection_type: ConnectionType, 
                      strength: float) -> BasinConnection:
        """Add connection to another basin"""
        connection = BasinConnection(
            basin_id=target_basin_id,
            connection_type=connection_type,
            connection_strength=strength
        )
        
        self.connected_basins.append(connection)
        
        # Update connection matrix
        if target_basin_id not in self.connection_matrix:
            self.connection_matrix[target_basin_id] = {}
        self.connection_matrix[target_basin_id][connection_type.value] = strength
        
        self.last_updated = datetime.utcnow()
        return connection
    
    def calculate_field_influence(self) -> float:
        """Calculate influence on neural field"""
        if self.current_state == BasinState.DORMANT:
            return 0.0
        
        # Base influence from activation and stability
        base_influence = self.current_activation * self.stability * 0.5
        
        # Factor in basin depth and width
        depth_factor = min(self.depth / 10.0, 1.0) * 0.3
        width_factor = min(self.width / 5.0, 1.0) * 0.2
        
        total_influence = base_influence + depth_factor + width_factor
        
        # Update neural field influence
        self.neural_field_influence.field_contribution = min(total_influence, 1.0)
        
        return self.neural_field_influence.field_contribution
    
    def update_consciousness_contribution(self) -> float:
        """Update contribution to consciousness emergence"""
        if self.current_state == BasinState.DORMANT:
            self.consciousness_contribution = 0.0
            return 0.0
        
        # Base contribution from activation and consciousness markers
        base_contribution = self.current_activation * 0.4
        
        # Factor in consciousness markers
        marker_contribution = len(self.consciousness_markers) * 0.1
        
        # Factor in emergence indicators
        emergence_contribution = sum(self.emergence_indicators.values()) * 0.1
        
        # Factor in metacognitive layer influence
        metacog_influence = self.layer_influences.get("metacognitive", 0.0) * 0.3
        
        total_contribution = (
            base_contribution + marker_contribution + 
            emergence_contribution + metacog_influence
        )
        
        self.consciousness_contribution = min(total_contribution, 1.0)
        return self.consciousness_contribution
    
    def is_stable(self, stability_threshold: float = 0.7) -> bool:
        """Check if basin is stable"""
        return self.stability >= stability_threshold
    
    def get_connected_basin_ids(self, connection_type: Optional[ConnectionType] = None) -> List[str]:
        """Get IDs of connected basins, optionally filtered by connection type"""
        if connection_type is None:
            return [conn.basin_id for conn in self.connected_basins]
        else:
            return [conn.basin_id for conn in self.connected_basins 
                   if conn.connection_type == connection_type]
    
    def calculate_total_influence(self) -> float:
        """Calculate total influence across all connected basins"""
        if self.current_state == BasinState.DORMANT:
            return 0.0
        
        # Sum influences from all connections
        total_influence = 0.0
        for connection in self.connected_basins:
            if connection.connection_type == ConnectionType.EXCITATORY:
                total_influence += connection.connection_strength * self.current_activation
            elif connection.connection_type == ConnectionType.INHIBITORY:
                total_influence -= connection.connection_strength * self.current_activation * 0.5
        
        return max(0.0, total_influence)
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        extra = "forbid"
