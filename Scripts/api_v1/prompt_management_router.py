"""
API router for prompt management.

This module provides API endpoints for managing prompt templates 
used with LLM services in the RAG pipeline.
"""
import os
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from pydantic import BaseModel, Field
import logging

from Scripts.llm.prompt_management import PromptTemplate, PromptLibrary, PromptVersion, PromptManager
from Scripts.llm.default_prompts import DefaultPromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API router
router = APIRouter(prefix="/prompt-management", tags=["Prompt Management"])

# Pydantic models for API
class PromptTemplateModel(BaseModel):
    """Model for prompt template."""
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    template: str = Field(..., description="Prompt template text with variable placeholders")
    variables: Optional[List[str]] = Field(None, description="List of variable names (auto-detected if None)")
    version: Optional[str] = Field("1.0", description="Template version")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "custom_qa_template",
                "description": "Custom template for question answering",
                "template": "Answer the question based on context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
                "variables": ["context", "question"],
                "version": "1.0"
            }
        }


class PromptVersionModel(BaseModel):
    """Model for prompt version."""
    version: str = Field(..., description="Version string (e.g., '1.0')")
    template: str = Field(..., description="Prompt template text")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "version": "1.0",
                "template": "Answer the question based on context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
                "created_at": "2023-06-01T10:00:00",
                "metrics": {"accuracy": 0.92, "latency_ms": 250}
            }
        }


class PromptLibraryModel(BaseModel):
    """Model for prompt library."""
    templates: List[PromptTemplateModel] = Field(..., description="List of templates")


class PromptTestRequest(BaseModel):
    """Model for prompt testing request."""
    template_name: str = Field(..., description="Template name to test")
    variables: Dict[str, str] = Field(..., description="Variable values for template")
    
    class Config:
        schema_extra = {
            "example": {
                "template_name": "qa_prompt",
                "variables": {
                    "context": "The first man on the moon was Neil Armstrong on July 20, 1969.",
                    "question": "Who was the first man on the moon?"
                }
            }
        }


class PromptComparisonRequest(BaseModel):
    """Model for prompt comparison request."""
    template_names: List[str] = Field(..., description="List of template names to compare")
    variables: Dict[str, str] = Field(..., description="Variable values for templates")
    
    class Config:
        schema_extra = {
            "example": {
                "template_names": ["qa_prompt", "custom_qa_template"],
                "variables": {
                    "context": "The first man on the moon was Neil Armstrong on July 20, 1969.",
                    "question": "Who was the first man on the moon?"
                }
            }
        }


# Initialize prompt manager
prompt_library = PromptLibrary()
prompt_manager = PromptManager(library=prompt_library)

# Make sure default templates are loaded
for name, template_data in DefaultPromptTemplates.TEMPLATES.items():
    prompt_template = PromptTemplate.from_dict(template_data)
    prompt_library.add_template(prompt_template)

# Helper dependency
def get_prompt_manager():
    return prompt_manager


@router.get("/templates", response_model=List[PromptTemplateModel], summary="List all templates")
async def list_templates():
    """List all available prompt templates."""
    templates = prompt_library.list_templates()
    return [template.to_dict() for template in templates]


@router.get("/templates/{name}", response_model=PromptTemplateModel, summary="Get template by name")
async def get_template(name: str = Path(..., description="Template name")):
    """Get a specific prompt template by name."""
    template = prompt_library.get_template(name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return template.to_dict()


@router.post("/templates", response_model=PromptTemplateModel, summary="Create new template")
async def create_template(template: PromptTemplateModel):
    """Create a new prompt template."""
    # Check if template already exists
    if prompt_library.get_template(template.name):
        raise HTTPException(status_code=400, detail=f"Template '{template.name}' already exists")
    
    # Create template
    prompt_template = PromptTemplate(
        template=template.template,
        name=template.name,
        description=template.description,
        variables=template.variables,
        version=template.version
    )
    
    # Add to library
    success = prompt_library.add_template(prompt_template)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add template")
    
    return prompt_template.to_dict()


@router.put("/templates/{name}", response_model=PromptTemplateModel, summary="Update template")
async def update_template(
    template: PromptTemplateModel,
    name: str = Path(..., description="Template name")
):
    """Update an existing prompt template."""
    # Check if template exists
    if not prompt_library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    # Create new template version
    prompt_template = PromptTemplate(
        template=template.template,
        name=name,
        description=template.description,
        variables=template.variables,
        version=template.version
    )
    
    # Update in library
    success = prompt_library.add_template(prompt_template)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update template")
    
    return prompt_template.to_dict()


@router.delete("/templates/{name}", summary="Delete template")
async def delete_template(name: str = Path(..., description="Template name")):
    """Delete a prompt template."""
    # Check if template exists
    if not prompt_library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    # Delete from library
    success = prompt_library.remove_template(name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete template")
    
    return {"message": f"Template '{name}' deleted successfully"}


@router.post("/test", summary="Test prompt template")
async def test_template(request: PromptTestRequest):
    """Test a prompt template with provided variables."""
    # Get template
    template = prompt_library.get_template(request.template_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")
    
    try:
        # Format template
        formatted = template.format(**request.variables)
        return {
            "formatted_prompt": formatted,
            "template_name": request.template_name,
            "variables": request.variables
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/versions/{name}", response_model=List[PromptVersionModel], summary="List template versions")
async def list_template_versions(
    name: str = Path(..., description="Template name")
):
    """List all versions of a specific template."""
    # Check if template exists
    if not prompt_library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    # Get versions
    versions = prompt_manager.get_template_versions(name)
    return [version.to_dict() for version in versions]


@router.post("/compare", summary="Compare multiple prompt templates")
async def compare_templates(request: PromptComparisonRequest):
    """Compare multiple prompt templates with the same variables."""
    results = {}
    
    for template_name in request.template_names:
        # Get template
        template = prompt_library.get_template(template_name)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        try:
            # Format template
            formatted = template.format(**request.variables)
            results[template_name] = formatted
        except ValueError as e:
            results[template_name] = f"Error: {str(e)}"
    
    return {
        "results": results,
        "variables": request.variables
    }


@router.post("/feedback", summary="Submit template feedback")
async def submit_template_feedback(
    template_name: str = Body(..., embed=True, description="Template name"),
    version: str = Body(..., embed=True, description="Template version"),
    user_id: Optional[str] = Body(None, embed=True, description="User ID"),
    feedback: Dict[str, Any] = Body(..., embed=True, description="Feedback data"),
):
    """Submit feedback for a specific template version."""
    # Check if template exists
    if not prompt_library.get_template(template_name):
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    # Record feedback
    success = prompt_manager.record_feedback(template_name, version, feedback, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    
    return {"message": "Feedback recorded successfully"}
"""
