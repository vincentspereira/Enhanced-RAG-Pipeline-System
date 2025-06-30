"""
API endpoints for the prompt management system.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from ...llm.prompt_system import (
    get_prompt_management_system,
    PromptTemplate, 
    PromptManagementSystem
)
from ...auth.dependencies import get_auth_manager_dependency
from ...auth.auth_manager import AuthUser, Permission # Removed AuthManager as it's implicitly handled by dependency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define models
class TemplateBase(BaseModel):
    """Base model for prompt templates."""
    name: str = Field(..., description="Name of the template")
    description: Optional[str] = Field(None, description="Description of the template")
    template: str = Field(..., description="Template string with variable placeholders")
    variables: List[str] = Field([], description="List of variable names")

class TemplateCreate(TemplateBase):
    """Model for creating a new template."""
    pass

class TemplateUpdate(BaseModel):
    """Model for updating an existing template."""
    description: Optional[str] = Field(None, description="New description")
    template: Optional[str] = Field(None, description="New template string")
    variables: Optional[List[str]] = Field(None, description="New variables list")

class TemplateResponse(TemplateBase):
    """Response model for templates."""
    version: str = Field(..., description="Template version")

class OptimizationSettings(BaseModel):
    """Settings for template optimization."""
    enabled: bool = Field(True, description="Whether optimization is enabled")
    targetMetric: str = Field("relevance", description="Target metric for optimization")
    iterations: int = Field(5, description="Number of optimization iterations")

class UsageStats(BaseModel):
    """Model for template usage statistics."""
    template: str = Field(..., description="Template name")
    usage_count: int = Field(..., description="Number of times the template was used")
    percentage_of_total: float = Field(..., description="Percentage of total usage")
    recent_usages: List[Dict[str, Any]] = Field([], description="Recent usages of the template")

class UsageStatsResponse(BaseModel):
    """Response model for usage statistics."""
    total_usage: int = Field(..., description="Total number of prompt generations")
    templates: List[Dict[str, Any]] = Field(..., description="Template usage statistics")

class PerformanceMetrics(BaseModel):
    """Model for template performance metrics."""
    template: str = Field(..., description="Template name")
    metrics_count: int = Field(..., description="Number of metric reports")
    average_metrics: Dict[str, float] = Field(..., description="Average metrics values")
    recent_metrics: List[Dict[str, Any]] = Field([], description="Recent metrics reports")

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    total_metrics_count: int = Field(..., description="Total number of metrics reports")
    templates: List[Dict[str, Any]] = Field(..., description="Template performance metrics")

# Create router
router = APIRouter(prefix="/prompts", tags=["prompts"])

# Helper function to get prompt management system
def get_pms() -> PromptManagementSystem:
    """Get the prompt management system."""
    return get_prompt_management_system()

# Template CRUD endpoints
@router.get("/templates", response_model=List[TemplateResponse])
async def list_templates(current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_READ))):
    """List all available prompt templates."""
    pms = get_pms()
    templates = []
    
    for name, template in pms.library.templates.items():
        templates.append(
            TemplateResponse(
                name=template.name,
                description=template.description,
                template=template.template,
                variables=template.variables,
                version=template.version
            )
        )
    
    return templates

@router.get("/templates/{name}", response_model=TemplateResponse)
async def get_template(
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_READ))
):
    """Get a specific prompt template by name."""
    pms = get_pms()
    template = pms.library.get_template(name)
    
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    return TemplateResponse(
        name=template.name,
        description=template.description,
        template=template.template,
        variables=template.variables,
        version=template.version
    )

@router.post("/templates", response_model=TemplateResponse)
async def create_template(
    template_data: TemplateCreate,
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE))
):
    """Create a new prompt template."""
    pms = get_pms()
    
    try:
        template = pms.create_template(
            template=template_data.template,
            name=template_data.name,
            description=template_data.description,
            variables=template_data.variables
        )
        
        return TemplateResponse(
            name=template.name,
            description=template.description,
            template=template.template,
            variables=template.variables,
            version=template.version
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/templates/{name}", response_model=TemplateResponse)
async def update_template(
    template_data: TemplateUpdate,
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE))
):
    """Update an existing prompt template."""
    pms = get_pms()
    
    try:
        template = pms.update_template(
            name=name,
            template=template_data.template,
            description=template_data.description,
            variables=template_data.variables
        )
        
        return TemplateResponse(
            name=template.name,
            description=template.description,
            template=template.template,
            variables=template.variables,
            version=template.version
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/templates/{name}", status_code=204)
async def delete_template(
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE))
):
    """Delete a prompt template."""
    pms = get_pms()
    
    if not pms.library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    success = pms.library.delete_template(name)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete template '{name}'")

# Template optimization endpoint
@router.post("/optimize/{name}", response_model=TemplateResponse)
async def optimize_template(
    optimization_settings: OptimizationSettings,
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE)) # Or a more specific PROMPT_OPTIMIZE perm
):
    """Optimize a prompt template based on metrics."""
    pms = get_pms()
    
    if not pms.library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    try:
        context = {
            "target_metric": optimization_settings.targetMetric,
            "iterations": optimization_settings.iterations
        }
        
        optimized_template = pms.optimize_prompt(name, context)
        template = pms.library.get_template(optimized_template)
        
        return TemplateResponse(
            name=template.name,
            description=template.description,
            template=template.template,
            variables=template.variables,
            version=template.version
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Analytics endpoints
@router.get("/usage", response_model=UsageStatsResponse)
async def get_usage_statistics(
    template_name: Optional[str] = Query(None, description="Filter by template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.ADMIN_READ))
):
    """Get usage statistics for prompt templates."""
    pms = get_pms()
    
    try:
        stats = pms.get_usage_stats(template_name)
        return stats
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage statistics: {str(e)}")

@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    template_name: Optional[str] = Query(None, description="Filter by template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.ADMIN_READ))
):
    """Get performance metrics for prompt templates."""
    pms = get_pms()
    
    try:
        metrics = pms.get_performance_metrics(template_name)
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# Track performance endpoint
@router.post("/performance/{name}")
async def track_template_performance(
    metrics: Dict[str, Any],
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE))
):
    """Track performance metrics for a prompt template."""
    pms = get_pms()
    
    if not pms.library.get_template(name):
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    
    try:
        pms.analytics.track_performance(name, metrics)
        return {"status": "success", "message": f"Performance metrics tracked for template '{name}'"}
    except Exception as e:
        logger.error(f"Error tracking performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track performance: {str(e)}")

# --- Endpoints migrated from prompt_management_router.py ---

class PromptTestRequest(BaseModel):
    """Model for prompt testing request."""
    template_name: str = Field(..., description="Template name to test")
    variables: Dict[str, str] = Field(..., description="Variable values for template")

    class Config:
        schema_extra = {
            "example": {
                "template_name": "qa_prompt", # Assuming a default template like 'qa_prompt' exists
                "variables": {
                    "context": "The first man on the moon was Neil Armstrong on July 20, 1969.",
                    "question": "Who was the first man on the moon?"
                }
            }
        }

class PromptVersionModel(BaseModel):
    """Model for prompt version."""
    version: str = Field(..., description="Version string (e.g., '1.0')")
    template: str = Field(..., description="Prompt template text")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")

@router.post("/test", summary="Test prompt template")
async def test_template(
    request: PromptTestRequest,
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_READ))
):
    """Test a prompt template with provided variables."""
    pms = get_pms()
    template = pms.library.get_template(request.template_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")

    try:
        formatted = template.format(**request.variables)
        # pms.analytics.track_usage(request.template_name, request.variables) # Usage is tracked by get_prompt
        return {
            "formatted_prompt": formatted,
            "template_name": request.template_name,
            "variables": request.variables
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/versions/{name}", response_model=List[PromptVersionModel], summary="List template versions")
async def list_template_versions(
    name: str = Path(..., description="Template name"),
    current_user: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_READ))
):
    """List all versions of a specific template."""
    pms = get_pms()
    if not pms.library.get_template(name): # Check if base template name exists
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")

    # Assuming PromptManagementSystem has a way to get versions, similar to PromptManager
    # This might require adding a method to PromptManagementSystem or accessing its internal PromptManager
    # For now, let's assume pms.get_template_versions(name) exists or can be implemented.
    # Placeholder:
    # versions_data = pms.get_template_versions(name) # This method needs to exist in PromptManagementSystem
    # For now, this might be a conceptual merge. The actual PromptManager in prompt_management_router
    # had this. PromptSystem in prompt_router.py would need to expose it.
    # Let's assume PromptManagementSystem has an attribute `manager` which is an instance of PromptManager
    if hasattr(pms, 'manager') and hasattr(pms.manager, 'get_template_versions'):
        versions = pms.manager.get_template_versions(name)
        return [version.to_dict() for version in versions]
    elif hasattr(pms, 'get_template_versions'): # If PMS itself has the method
        versions = pms.get_template_versions(name)
        return [version.to_dict() for version in versions]
    else:
        # Fallback or raise error if versioning info isn't directly accessible via PMS
        logger.warning(f"Versioning info for template '{name}' not directly accessible via PromptManagementSystem.")
        # Returning current version as the only version for now
        template = pms.library.get_template(name)
        return [PromptVersionModel(version=template.version, template=template.template, created_at=None, metrics={})]


@router.post("/feedback", summary="Submit template feedback")
async def submit_template_feedback(
    template_name: str = Body(..., embed=True, description="Template name"),
    version: str = Body(..., embed=True, description="Template version"),
    user_id: Optional[str] = Body(None, embed=True, description="User ID from request or auth"),
    feedback: Dict[str, Any] = Body(..., embed=True, description="Feedback data"),
    current_user_auth: AuthUser = Depends(get_auth_manager_dependency().require_permission(Permission.API_WRITE)) # Renamed, applied new auth
):
    """Submit feedback for a specific template version."""
    pms = get_pms()
    if not pms.library.get_template(template_name):
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

    # Use authenticated user ID if user_id is not provided in body
    effective_user_id = user_id or (current_user_auth.id if hasattr(current_user_auth, 'id') else "anonymous")

    # Assuming PromptManagementSystem has a way to record feedback
    # This might require adding a method to PromptManagementSystem or accessing its internal PromptManager
    if hasattr(pms, 'manager') and hasattr(pms.manager, 'record_feedback'):
        success = pms.manager.record_feedback(template_name, version, feedback, effective_user_id)
    elif hasattr(pms, 'record_feedback'):
        success = pms.record_feedback(template_name, version, feedback, effective_user_id)
    else:
        logger.error(f"Feedback recording mechanism not found in PromptManagementSystem for {template_name}")
        raise HTTPException(status_code=501, detail="Feedback recording not implemented in PromptManagementSystem")

    if not success:
        raise HTTPException(status_code=500, detail="Failed to record feedback")

    return {"message": "Feedback recorded successfully"}
