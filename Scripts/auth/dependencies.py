from fastapi import Request, Depends
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Scripts.auth.auth_manager import AuthManager

def get_auth_manager_dependency(request: Request) -> 'AuthManager':
    """
    Dependency to get the AuthManager instance from the FastAPI app state.
    Routers will depend on this to get the auth_manager,
    and then use auth_manager.require_permission(Permission.XYZ)
    """
    if not hasattr(request.app.state, 'auth_manager'):
        # This should not happen if startup events are correctly run
        raise RuntimeError("AuthManager not found in app.state. Ensure it's initialized in FastAPI startup.")
    return request.app.state.auth_manager
