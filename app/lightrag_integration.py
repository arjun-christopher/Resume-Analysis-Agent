# app/lightrag_integration.py - LightRAG Integration Module
"""
Integration module for LightRAG package while maintaining compatibility 
with existing fast_semantic_rag system. Provides hybrid RAG capabilities.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
from dataclasses import dataclass

# LightRAG imports
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    _HAS_LIGHTRAG = True
except ImportError:
    _HAS_LIGHTRAG = False
    LightRAG = None
    QueryParam = None

from fast_semantic_rag import FastSemanticRAG, FastRAGConfig


@dataclass
class HybridRAGConfig:
    """Configuration for hybrid RAG system combining FastSemanticRAG and LightRAG"""
    
    # LightRAG specific settings
    enable_lightrag: bool = True
    lightrag_working_dir: str = "data/lightrag"
    lightrag_chunk_size: int = 1200
    lightrag_chunk_overlap: int = 100
    
    # FastSemanticRAG settings (fallback and complementary)
    enable_fast_rag: bool = True
    fast_rag_config: Optional[FastRAGConfig] = None
    
    # Hybrid operation modes
    query_mode: str = "hybrid"  # "lightrag_only", "fast_rag_only", "hybrid"
    lightrag_weight: float = 0.7  # Weight for LightRAG results in hybrid mode
    
    # Performance settings
    max_processing_time: float = 30.0  # Maximum processing time per query
    enable_caching: bool = True
    
    # LLM settings (for LightRAG)
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1


class HybridRAGSystem:
    """
    Hybrid RAG system that combines LightRAG's advanced knowledge graph capabilities
    with FastSemanticRAG's speed optimizations for resume analysis.
    """
    
    def __init__(self, config: HybridRAGConfig = None, storage_path: str = "data/hybrid_rag"):
        self.config = config or HybridRAGConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.lightrag_system = None
        self.fast_rag_system = None
        self.is_initialized = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize both RAG systems"""
        try:
            # Initialize LightRAG if available and enabled
            if _HAS_LIGHTRAG and self.config.enable_lightrag:
                await self._initialize_lightrag()
            
            # Initialize FastSemanticRAG as fallback/complement
            if self.config.enable_fast_rag:
                self._initialize_fast_rag()
            
            self.is_initialized = True
            self.logger.info("Hybrid RAG system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Hybrid RAG system: {e}")
            # Fall back to FastSemanticRAG only
            if not self.fast_rag_system and self.config.enable_fast_rag:
                self._initialize_fast_rag()
            
    async def _initialize_lightrag(self):
        """Initialize LightRAG system"""
        if not _HAS_LIGHTRAG:
            self.logger.warning("LightRAG not available, skipping initialization")
            return
            
        try:
            # Get API key from config or environment
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OpenAI API key not found, LightRAG may not work properly")
            
            # Create LightRAG instance
            self.lightrag_system = LightRAG(
                working_dir=self.config.lightrag_working_dir,
                chunk_token_size=self.config.lightrag_chunk_size,
                chunk_overlap_token_size=self.config.lightrag_chunk_overlap,
                llm_model_func=self._create_llm_func(api_key),
                embedding_func=EmbeddingFunc(
                    embedding_dim=1536,  # text-embedding-3-small dimension
                    func=lambda texts: openai_embed(
                        texts,
                        model=self.config.embedding_model,
                        api_key=api_key
                    )
                )
            )
            
            # Initialize storage and pipeline
            await self.lightrag_system.initialize_storages()
            await initialize_pipeline_status()
            
            self.logger.info("LightRAG system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LightRAG: {e}")
            self.lightrag_system = None
    
    def _create_llm_func(self, api_key: str):
        """Create LLM function for LightRAG"""
        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                self.config.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                temperature=self.config.temperature,
                **kwargs
            )
        return llm_func
    
    def _initialize_fast_rag(self):
        """Initialize FastSemanticRAG system"""
        try:
            fast_config = self.config.fast_rag_config or FastRAGConfig()
            self.fast_rag_system = FastSemanticRAG(
                config=fast_config,
                storage_path=str(self.storage_path / "fast_rag")
            )
            self.logger.info("FastSemanticRAG system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FastSemanticRAG: {e}")
            self.fast_rag_system = None
    
    async def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to both RAG systems"""
        results = {"lightrag": None, "fast_rag": None, "success": False}
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Add to LightRAG
            if self.lightrag_system:
                try:
                    for i, doc in enumerate(documents):
                        await self.lightrag_system.ainsert(doc)
                    results["lightrag"] = {"documents_added": len(documents)}
                    self.logger.info(f"Added {len(documents)} documents to LightRAG")
                except Exception as e:
                    self.logger.error(f"Failed to add documents to LightRAG: {e}")
            
            # Add to FastSemanticRAG
            if self.fast_rag_system:
                try:
                    result = self.fast_rag_system.add_documents(documents, metadata)
                    results["fast_rag"] = result
                    self.logger.info(f"Added {len(documents)} documents to FastSemanticRAG")
                except Exception as e:
                    self.logger.error(f"Failed to add documents to FastSemanticRAG: {e}")
            
            results["success"] = results["lightrag"] is not None or results["fast_rag"] is not None
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            
        return results
    
    async def query(self, question: str, mode: str = None) -> Dict[str, Any]:
        """
        Query the hybrid RAG system with intelligent routing and result combination
        """
        if not self.is_initialized:
            await self.initialize()
        
        query_mode = mode or self.config.query_mode
        results = {
            "answer": "",
            "sources": [],
            "processing_time": 0.0,
            "method_used": query_mode,
            "lightrag_result": None,
            "fast_rag_result": None
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if query_mode == "lightrag_only" and self.lightrag_system:
                results.update(await self._query_lightrag(question))
                
            elif query_mode == "fast_rag_only" and self.fast_rag_system:
                results.update(await self._query_fast_rag(question))
                
            elif query_mode == "hybrid":
                results.update(await self._query_hybrid(question))
                
            else:
                # Fallback to available system
                if self.lightrag_system:
                    results.update(await self._query_lightrag(question))
                elif self.fast_rag_system:
                    results.update(await self._query_fast_rag(question))
                else:
                    results["answer"] = "No RAG system available"
        
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            results["answer"] = f"Query failed: {str(e)}"
        
        results["processing_time"] = asyncio.get_event_loop().time() - start_time
        return results
    
    async def _query_lightrag(self, question: str) -> Dict[str, Any]:
        """Query LightRAG system"""
        try:
            # Use hybrid mode for best results
            result = await self.lightrag_system.aquery(
                question,
                param=QueryParam(mode="hybrid")
            )
            
            return {
                "answer": result,
                "method_used": "lightrag",
                "lightrag_result": result
            }
        except Exception as e:
            self.logger.error(f"LightRAG query failed: {e}")
            return {"answer": f"LightRAG query failed: {str(e)}"}
    
    async def _query_fast_rag(self, question: str) -> Dict[str, Any]:
        """Query FastSemanticRAG system"""
        try:
            result = self.fast_rag_system.query(question)
            
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "method_used": "fast_rag",
                "fast_rag_result": result
            }
        except Exception as e:
            self.logger.error(f"FastSemanticRAG query failed: {e}")
            return {"answer": f"FastSemanticRAG query failed: {str(e)}"}
    
    async def _query_hybrid(self, question: str) -> Dict[str, Any]:
        """Query both systems and combine results intelligently"""
        lightrag_result = None
        fast_rag_result = None
        
        # Query both systems concurrently
        tasks = []
        if self.lightrag_system:
            tasks.append(self._query_lightrag(question))
        if self.fast_rag_system:
            tasks.append(self._query_fast_rag(question))
        
        if not tasks:
            return {"answer": "No RAG systems available"}
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result.get("method_used") == "lightrag":
                    lightrag_result = result
                elif result.get("method_used") == "fast_rag":
                    fast_rag_result = result
            
            # Combine results intelligently
            combined_answer = self._combine_answers(lightrag_result, fast_rag_result, question)
            
            return {
                "answer": combined_answer,
                "sources": fast_rag_result.get("sources", []) if fast_rag_result else [],
                "method_used": "hybrid",
                "lightrag_result": lightrag_result,
                "fast_rag_result": fast_rag_result
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid query failed: {e}")
            return {"answer": f"Hybrid query failed: {str(e)}"}
    
    def _combine_answers(self, lightrag_result: Dict, fast_rag_result: Dict, question: str) -> str:
        """Intelligently combine answers from both systems"""
        
        # If only one system has results, use that
        if lightrag_result and not fast_rag_result:
            return lightrag_result.get("answer", "")
        elif fast_rag_result and not lightrag_result:
            return fast_rag_result.get("answer", "")
        elif not lightrag_result and not fast_rag_result:
            return "No results found"
        
        # Both systems have results - combine them
        lightrag_answer = lightrag_result.get("answer", "")
        fast_rag_answer = fast_rag_result.get("answer", "")
        
        # Determine which system is better for this query type
        if any(keyword in question.lower() for keyword in ["name", "contact", "email", "phone"]):
            # For contact information, FastRAG might be more precise
            primary_answer = fast_rag_answer
            secondary_answer = lightrag_answer
        elif any(keyword in question.lower() for keyword in ["experience", "relationship", "overview", "summary"]):
            # For complex queries, LightRAG might be better
            primary_answer = lightrag_answer
            secondary_answer = fast_rag_answer
        else:
            # Use configuration weights
            if self.config.lightrag_weight > 0.5:
                primary_answer = lightrag_answer
                secondary_answer = fast_rag_answer
            else:
                primary_answer = fast_rag_answer
                secondary_answer = lightrag_answer
        
        # Combine the answers
        if primary_answer and secondary_answer:
            if len(primary_answer) > len(secondary_answer) * 2:
                return primary_answer  # Primary is much more detailed
            else:
                return f"{primary_answer}\n\nAdditional insights:\n{secondary_answer}"
        else:
            return primary_answer or secondary_answer
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid system"""
        stats = {
            "lightrag_available": self.lightrag_system is not None,
            "fast_rag_available": self.fast_rag_system is not None,
            "is_initialized": self.is_initialized,
            "query_mode": self.config.query_mode
        }
        
        if self.fast_rag_system:
            stats["fast_rag_stats"] = self.fast_rag_system.get_system_stats()
        
        return stats


def create_hybrid_rag_system(storage_path: str = "data/hybrid_rag", **config_kwargs) -> HybridRAGSystem:
    """Factory function to create a configured hybrid RAG system"""
    config = HybridRAGConfig(**config_kwargs)
    return HybridRAGSystem(config, storage_path)


async def initialize_hybrid_rag_system(storage_path: str = "data/hybrid_rag", **config_kwargs) -> HybridRAGSystem:
    """Factory function to create and initialize a hybrid RAG system"""
    system = create_hybrid_rag_system(storage_path, **config_kwargs)
    await system.initialize()
    return system