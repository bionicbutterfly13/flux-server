"""
Ollama Integration Service for Ultra-Granular Document Processing
================================================================

Provides local LLM capabilities through Ollama for:
- Concept extraction and analysis
- Relationship mapping
- Text embeddings
- Model health monitoring and fallback

Implements Spec-022 requirements for Ollama-powered consciousness processing.
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import httpx
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of Ollama models for different tasks"""
    PRIMARY = "primary"        # Main reasoning model (qwen2.5:14b)
    FAST = "fast"             # Quick processing (llama3.2:3b)
    EMBEDDING = "embedding"    # Text embeddings (nomic-embed-text)
    SPECIALIZED = "specialized" # Domain-specific models

class ModelStatus(Enum):
    """Model availability status"""
    HEALTHY = "healthy"
    SLOW = "slow"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class ModelConfig:
    """Configuration for an Ollama model"""
    name: str
    type: ModelType
    endpoint: str = "http://localhost:11434"
    timeout: float = 90.0
    max_retries: int = 3
    fallback_model: Optional[str] = None
    
    # Performance thresholds
    healthy_response_time: float = 5.0
    slow_response_time: float = 15.0
    
    # Model-specific parameters
    temperature: float = 0.1
    top_p: float = 0.9
    context_length: int = 4096

@dataclass
class ModelHealthStatus:
    """Health status of a model"""
    model_name: str
    status: ModelStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0

@dataclass
class ConceptExtractionRequest:
    """Request for concept extraction"""
    text: str
    granularity_level: int  # 1-5 for ultra-granular processing
    domain_focus: List[str] = field(default_factory=lambda: ["neuroscience", "ai"])
    extraction_type: str = "atomic_concepts"  # atomic_concepts, relationships, composite, etc.
    context: Optional[Dict[str, Any]] = None

@dataclass
class ConceptExtractionResponse:
    """Response from concept extraction"""
    success: bool
    concepts: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    processing_time: float
    confidence_score: float
    model_used: str
    error_message: Optional[str] = None

class OllamaModelManager:
    """Manages Ollama models for ultra-granular processing"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.health_status: Dict[str, ModelHealthStatus] = {}
        self.client = httpx.AsyncClient(timeout=120.0)
        
        # Initialize default model configurations
        self._setup_default_models()
        
        # Performance tracking
        self.request_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
    def _setup_default_models(self):
        """Setup default model configurations for ultra-granular processing"""
        
        # Primary reasoning model for complex concept analysis (now available)
        self.models["qwen2.5:14b"] = ModelConfig(
            name="qwen2.5:14b",
            type=ModelType.PRIMARY,
            timeout=60.0,  # Longer timeout for large model
            fallback_model="qwen2.5:7b",
            temperature=0.1,
            context_length=8192
        )
        
        # Fast model for simple extractions
        self.models["qwen2.5:7b"] = ModelConfig(
            name="qwen2.5:7b", 
            type=ModelType.FAST,
            timeout=90.0,
            fallback_model="llama3.2:3b",
            temperature=0.1,
            context_length=4096
        )
        
        # Backup fast model
        self.models["llama3.2:3b"] = ModelConfig(
            name="llama3.2:3b",
            type=ModelType.FAST,
            timeout=20.0,
            temperature=0.1,
            context_length=2048
        )
        
        # Embedding model for vector generation
        self.models["nomic-embed-text"] = ModelConfig(
            name="nomic-embed-text",
            type=ModelType.EMBEDDING,
            timeout=15.0,
            temperature=0.0,
            context_length=2048
        )
        
        logger.info(f"Configured {len(self.models)} Ollama models")
    
    async def check_model_health(self, model_name: str) -> ModelHealthStatus:
        """Check health status of a specific model"""
        start_time = time.time()
        
        try:
            model_config = self.models[model_name]
            
            # For embedding models, test embeddings endpoint
            if model_config.type == ModelType.EMBEDDING:
                response = await self.client.post(
                    f"{model_config.endpoint}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": "test"
                    },
                    timeout=model_config.timeout
                )
            else:
                # For other models, test generation endpoint
                response = await self.client.post(
                    f"{model_config.endpoint}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Test",
                        "stream": False,
                        "options": {"num_predict": 5}
                    },
                    timeout=model_config.timeout
                )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Determine status based on response time
                config = self.models[model_name]
                if response_time <= config.healthy_response_time:
                    status = ModelStatus.HEALTHY
                elif response_time <= config.slow_response_time:
                    status = ModelStatus.SLOW
                else:
                    status = ModelStatus.ERROR
                
                health = ModelHealthStatus(
                    model_name=model_name,
                    status=status,
                    last_check=datetime.now(),
                    response_time=response_time
                )
                
                # Update success rate
                if model_name in self.health_status:
                    prev_health = self.health_status[model_name]
                    prev_health.total_requests += 1
                    if status != ModelStatus.ERROR:
                        health.success_rate = (prev_health.success_rate * (prev_health.total_requests - 1)) / prev_health.total_requests
                    else:
                        prev_health.failed_requests += 1
                        health.success_rate = 1.0 - (prev_health.failed_requests / prev_health.total_requests)
                    health.total_requests = prev_health.total_requests
                    health.failed_requests = prev_health.failed_requests
                
                self.health_status[model_name] = health
                return health
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            health = ModelHealthStatus(
                model_name=model_name,
                status=ModelStatus.ERROR,
                last_check=datetime.now(),
                response_time=time.time() - start_time,
                error_message=str(e)
            )
            
            # Update failure statistics
            if model_name in self.health_status:
                prev_health = self.health_status[model_name]
                prev_health.total_requests += 1
                prev_health.failed_requests += 1
                health.success_rate = 1.0 - (prev_health.failed_requests / prev_health.total_requests)
                health.total_requests = prev_health.total_requests
                health.failed_requests = prev_health.failed_requests
            
            self.health_status[model_name] = health
            logger.error(f"Model {model_name} health check failed: {e}")
            return health
    
    async def check_all_models_health(self) -> Dict[str, ModelHealthStatus]:
        """Check health of all configured models"""
        logger.info("Checking health of all Ollama models...")
        
        tasks = [self.check_model_health(model_name) for model_name in self.models.keys()]
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for model_name, health in zip(self.models.keys(), health_results):
            if isinstance(health, Exception):
                results[model_name] = ModelHealthStatus(
                    model_name=model_name,
                    status=ModelStatus.ERROR,
                    last_check=datetime.now(),
                    response_time=0.0,
                    error_message=str(health)
                )
            else:
                results[model_name] = health
        
        logger.info(f"Health check completed for {len(results)} models")
        return results
    
    def get_best_model_for_task(self, task_type: str, complexity: str = "medium") -> str:
        """Select best available model for a specific task"""
        
        # Define task-specific model preferences
        task_preferences = {
            "atomic_concept_extraction": {
                "high": ["qwen2.5:14b", "qwen2.5:7b", "llama3.2:3b"],
                "medium": ["qwen2.5:7b", "qwen2.5:14b", "llama3.2:3b"],
                "low": ["llama3.2:3b", "qwen2.5:7b"]
            },
            "relationship_mapping": {
                "high": ["qwen2.5:14b", "qwen2.5:7b"],
                "medium": ["qwen2.5:14b", "qwen2.5:7b"],
                "low": ["qwen2.5:7b", "llama3.2:3b"]
            },
            "composite_concept_assembly": {
                "high": ["qwen2.5:14b"],
                "medium": ["qwen2.5:14b", "qwen2.5:7b"],
                "low": ["qwen2.5:7b"]
            },
            "narrative_analysis": {
                "high": ["qwen2.5:14b"],
                "medium": ["qwen2.5:14b"],
                "low": ["qwen2.5:14b", "qwen2.5:7b"]
            },
            "embedding_generation": {
                "high": ["nomic-embed-text"],
                "medium": ["nomic-embed-text"],
                "low": ["nomic-embed-text"]
            }
        }
        
        preferred_models = task_preferences.get(task_type, {}).get(complexity, ["qwen2.5:7b"])
        
        # Select first healthy model from preferences
        for model_name in preferred_models:
            if model_name in self.health_status:
                health = self.health_status[model_name]
                if health.status in [ModelStatus.HEALTHY, ModelStatus.SLOW]:
                    logger.info(f"Selected {model_name} for {task_type} (complexity: {complexity})")
                    return model_name
        
        # Fallback to any healthy model
        for model_name, health in self.health_status.items():
            if health.status in [ModelStatus.HEALTHY, ModelStatus.SLOW]:
                logger.warning(f"Using fallback model {model_name} for {task_type}")
                return model_name
        
        # Last resort - return primary model even if unhealthy
        logger.error(f"No healthy models available for {task_type}, using qwen2.5:7b as last resort")
        return "qwen2.5:7b"
    
    async def generate_text(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using specified Ollama model"""
        start_time = time.time()
        
        try:
            model_config = self.models.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Prepare request
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", model_config.temperature),
                    "top_p": kwargs.get("top_p", model_config.top_p),
                    "num_predict": kwargs.get("max_tokens", 512),
                }
            }
            
            # Make request
            response = await self.client.post(
                f"{model_config.endpoint}/api/generate",
                json=request_data,
                timeout=model_config.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Track performance
                self._track_request(model_name, "generate", processing_time, True)
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model_name,
                    "processing_time": processing_time,
                    "total_duration": result.get("total_duration", 0) / 1e9,  # Convert to seconds
                    "load_duration": result.get("load_duration", 0) / 1e9,
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0)
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._track_request(model_name, "generate", processing_time, False, str(e))
            
            logger.error(f"Text generation failed with {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model_name,
                "processing_time": processing_time
            }
    
    async def generate_embeddings(self, text: str, model_name: str = "nomic-embed-text") -> Dict[str, Any]:
        """Generate embeddings using Ollama embedding model"""
        start_time = time.time()
        
        try:
            model_config = self.models.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown embedding model: {model_name}")
            
            response = await self.client.post(
                f"{model_config.endpoint}/api/embeddings",
                json={
                    "model": model_name,
                    "prompt": text
                },
                timeout=model_config.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                embeddings = np.array(result.get("embedding", []))
                
                self._track_request(model_name, "embeddings", processing_time, True)
                
                return {
                    "success": True,
                    "embeddings": embeddings,
                    "model": model_name,
                    "processing_time": processing_time,
                    "dimensions": len(embeddings)
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._track_request(model_name, "embeddings", processing_time, False, str(e))
            
            logger.error(f"Embedding generation failed with {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model_name,
                "processing_time": processing_time
            }
    
    def _track_request(self, model_name: str, request_type: str, processing_time: float, 
                      success: bool, error: Optional[str] = None):
        """Track request performance and history"""
        request_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "type": request_type,
            "processing_time": processing_time,
            "success": success,
            "error": error
        }
        
        self.request_history.append(request_record)
        
        # Maintain history size
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        metrics = {
            "total_requests": len(self.request_history),
            "models": {}
        }
        
        # Per-model metrics
        for model_name in self.models.keys():
            model_requests = [r for r in self.request_history if r["model"] == model_name]
            if model_requests:
                successful_requests = [r for r in model_requests if r["success"]]
                
                metrics["models"][model_name] = {
                    "total_requests": len(model_requests),
                    "successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / len(model_requests),
                    "avg_processing_time": sum(r["processing_time"] for r in successful_requests) / len(successful_requests) if successful_requests else 0,
                    "health_status": self.health_status.get(model_name, {}).status.value if model_name in self.health_status else "unknown"
                }
        
        return metrics
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class OllamaConceptExtractor:
    """Specialized concept extraction using Ollama models"""
    
    def __init__(self, model_manager: OllamaModelManager):
        self.model_manager = model_manager
        
        # Neuroscience/AI domain-specific prompts
        self.domain_prompts = {
            "neuroscience": {
                "atomic_concepts": """Extract individual neuroscience concepts from this text. Focus on:
- Neurological structures (neurons, synapses, brain regions)
- Processes (action potentials, neurotransmission, plasticity)
- Phenomena (consciousness, memory, perception)
- Measurement terms (spike rates, connectivity, activation)

Text: {text}

Return each concept as: CONCEPT: [name] | DEFINITION: [brief definition] | CONFIDENCE: [0-1]""",
                
                "relationships": """Identify relationships between neuroscience concepts in this text. Focus on:
- Causal relationships (A causes B, A influences B)
- Structural relationships (A contains B, A connects to B)
- Functional relationships (A processes B, A regulates B)
- Temporal relationships (A precedes B, A follows B)

Text: {text}

Return each relationship as: RELATION: [concept1] -> [relationship_type] -> [concept2] | STRENGTH: [0-1] | CONFIDENCE: [0-1]"""
            },
            "ai": {
                "atomic_concepts": """Extract individual AI/machine learning concepts from this text. Focus on:
- Algorithms (neural networks, transformers, reinforcement learning)
- Architectures (CNNs, RNNs, attention mechanisms)
- Processes (training, inference, optimization)
- Metrics (accuracy, loss, performance measures)

Text: {text}

Return each concept as: CONCEPT: [name] | DEFINITION: [brief definition] | CONFIDENCE: [0-1]""",
                
                "relationships": """Identify relationships between AI concepts in this text. Focus on:
- Implementation relationships (A implements B, A uses B)
- Performance relationships (A improves B, A optimizes B)
- Architectural relationships (A contains B, A builds on B)
- Training relationships (A trains B, A learns from B)

Text: {text}

Return each relationship as: RELATION: [concept1] -> [relationship_type] -> [concept2] | STRENGTH: [0-1] | CONFIDENCE: [0-1]"""
            }
        }
    
    async def extract_atomic_concepts(self, request: ConceptExtractionRequest) -> ConceptExtractionResponse:
        """Extract atomic concepts (Level 1) using Ollama"""
        start_time = time.time()
        
        # Select best model for task
        model_name = self.model_manager.get_best_model_for_task("atomic_concept_extraction", "medium")
        
        # Build domain-specific prompt
        concepts = []
        relationships = []
        
        for domain in request.domain_focus:
            if domain in self.domain_prompts:
                prompt = self.domain_prompts[domain]["atomic_concepts"].format(text=request.text)
                
                # Generate concepts
                result = await self.model_manager.generate_text(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.1
                )
                
                if result["success"]:
                    # Parse extracted concepts
                    extracted = self._parse_concept_response(result["response"], domain)
                    concepts.extend(extracted)
        
        processing_time = time.time() - start_time
        
        return ConceptExtractionResponse(
            success=len(concepts) > 0,
            concepts=concepts,
            relationships=relationships,
            processing_time=processing_time,
            confidence_score=sum(c.get("confidence", 0.5) for c in concepts) / len(concepts) if concepts else 0.0,
            model_used=model_name
        )
    
    async def extract_relationships(self, request: ConceptExtractionRequest) -> ConceptExtractionResponse:
        """Extract concept relationships (Level 2) using Ollama"""
        start_time = time.time()
        
        model_name = self.model_manager.get_best_model_for_task("relationship_mapping", "high")
        
        concepts = []
        relationships = []
        
        for domain in request.domain_focus:
            if domain in self.domain_prompts:
                prompt = self.domain_prompts[domain]["relationships"].format(text=request.text)
                
                result = await self.model_manager.generate_text(
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.1
                )
                
                if result["success"]:
                    extracted = self._parse_relationship_response(result["response"], domain)
                    relationships.extend(extracted)
        
        processing_time = time.time() - start_time
        
        return ConceptExtractionResponse(
            success=len(relationships) > 0,
            concepts=concepts,
            relationships=relationships,
            processing_time=processing_time,
            confidence_score=sum(r.get("confidence", 0.5) for r in relationships) / len(relationships) if relationships else 0.0,
            model_used=model_name
        )
    
    def _parse_concept_response(self, response: str, domain: str) -> List[Dict[str, Any]]:
        """Parse concept extraction response from Ollama"""
        concepts = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('CONCEPT:'):
                try:
                    # Parse: CONCEPT: [name] | DEFINITION: [definition] | CONFIDENCE: [score]
                    parts = line.split(' | ')
                    name = parts[0].replace('CONCEPT:', '').strip()
                    definition = parts[1].replace('DEFINITION:', '').strip() if len(parts) > 1 else ""
                    confidence = float(parts[2].replace('CONFIDENCE:', '').strip()) if len(parts) > 2 else 0.7
                    
                    concepts.append({
                        "name": name,
                        "definition": definition,
                        "domain": domain,
                        "confidence": confidence,
                        "type": "atomic_concept"
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse concept line: {line}, error: {e}")
        
        return concepts
    
    def _parse_relationship_response(self, response: str, domain: str) -> List[Dict[str, Any]]:
        """Parse relationship extraction response from Ollama"""
        relationships = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('RELATION:'):
                try:
                    # Parse: RELATION: [concept1] -> [type] -> [concept2] | STRENGTH: [score] | CONFIDENCE: [score]
                    parts = line.split(' | ')
                    relation_part = parts[0].replace('RELATION:', '').strip()
                    
                    # Extract relationship components
                    if ' -> ' in relation_part:
                        relation_components = relation_part.split(' -> ')
                        if len(relation_components) >= 3:
                            source_concept = relation_components[0].strip()
                            relationship_type = relation_components[1].strip()
                            target_concept = relation_components[2].strip()
                            
                            strength = float(parts[1].replace('STRENGTH:', '').strip()) if len(parts) > 1 else 0.7
                            confidence = float(parts[2].replace('CONFIDENCE:', '').strip()) if len(parts) > 2 else 0.7
                            
                            relationships.append({
                                "source_concept": source_concept,
                                "target_concept": target_concept,
                                "relationship_type": relationship_type,
                                "strength": strength,
                                "confidence": confidence,
                                "domain": domain
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse relationship line: {line}, error: {e}")
        
        return relationships


# Global instance
ollama_manager = OllamaModelManager()
ollama_extractor = OllamaConceptExtractor(ollama_manager)


# Initialization function
async def initialize_ollama_service():
    """Initialize Ollama service and check model health"""
    logger.info("Initializing Ollama integration service...")
    
    # Check health of all models
    health_status = await ollama_manager.check_all_models_health()
    
    healthy_models = [name for name, status in health_status.items() 
                     if status.status in [ModelStatus.HEALTHY, ModelStatus.SLOW]]
    
    if healthy_models:
        logger.info(f"✅ Ollama service initialized with {len(healthy_models)} healthy models: {healthy_models}")
        return True
    else:
        logger.error("❌ No healthy Ollama models available")
        return False


# Cleanup function
async def cleanup_ollama_service():
    """Cleanup Ollama service resources"""
    await ollama_manager.close()
    logger.info("Ollama service cleanup completed")


# Test function
async def test_ollama_integration():
    """Test Ollama integration with sample content"""
    logger.info("🧪 Testing Ollama integration...")
    
    # Initialize service
    initialized = await initialize_ollama_service()
    if not initialized:
        logger.error("Failed to initialize Ollama service")
        return False
    
    # Test concept extraction
    test_request = ConceptExtractionRequest(
        text="Neural networks learn through backpropagation, adjusting synaptic weights to minimize error. This process mimics synaptic plasticity in biological neurons.",
        granularity_level=1,
        domain_focus=["neuroscience", "ai"],
        extraction_type="atomic_concepts"
    )
    
    # Test atomic concept extraction
    concept_result = await ollama_extractor.extract_atomic_concepts(test_request)
    if concept_result.success:
        logger.info(f"✅ Extracted {len(concept_result.concepts)} concepts in {concept_result.processing_time:.2f}s")
        for concept in concept_result.concepts[:3]:  # Show first 3
            logger.info(f"  - {concept['name']} (confidence: {concept['confidence']:.2f})")
    else:
        logger.error("❌ Concept extraction failed")
    
    # Test relationship extraction
    relationship_result = await ollama_extractor.extract_relationships(test_request)
    if relationship_result.success:
        logger.info(f"✅ Extracted {len(relationship_result.relationships)} relationships in {relationship_result.processing_time:.2f}s")
        for rel in relationship_result.relationships[:2]:  # Show first 2
            logger.info(f"  - {rel['source_concept']} -> {rel['relationship_type']} -> {rel['target_concept']}")
    else:
        logger.error("❌ Relationship extraction failed")
    
    # Test embeddings
    embedding_result = await ollama_manager.generate_embeddings("consciousness and neural networks")
    if embedding_result["success"]:
        logger.info(f"✅ Generated embeddings: {embedding_result['dimensions']} dimensions in {embedding_result['processing_time']:.2f}s")
    else:
        logger.error("❌ Embedding generation failed")
    
    # Show performance metrics
    metrics = ollama_manager.get_performance_metrics()
    logger.info(f"📊 Performance: {metrics['total_requests']} total requests")
    
    await cleanup_ollama_service()
    logger.info("🎉 Ollama integration test completed!")
    
    return concept_result.success and relationship_result.success and embedding_result["success"]


if __name__ == "__main__":
    # Run test when executed directly
    asyncio.run(test_ollama_integration())