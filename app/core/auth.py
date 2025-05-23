from typing import Optional, Tuple
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserAPIKey
from app.services.auth_service import APIKeyService

security = HTTPBearer(auto_error=False)


async def get_current_user_from_session(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Get current user from session (for web interface)."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    
    user = db.query(User).filter(
        User.id == user_id,
        User.is_active == True
    ).first()
    
    return user


async def get_current_user_from_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Tuple[User, UserAPIKey]]:
    """Get current user from API key (for API access)."""
    if not credentials:
        return None
    
    # Verify API key
    result = await APIKeyService.verify_api_key(credentials.credentials)
    if not result:
        return None
    
    api_key, user = result
    return user, api_key


async def get_current_user_flexible(
    request: Request, 
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from either session or API key.
    Supports both web interface and API access.
    """
    # Try API key authentication first
    if credentials:
        result = await APIKeyService.verify_api_key(credentials.credentials)
        if result:
            api_key, user = result
            return user
    
    # Fall back to session authentication
    user_id = request.session.get("user_id")
    if user_id:
        user = db.query(User).filter(
            User.id == user_id,
            User.is_active == True
        ).first()
        if user:
            return user
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a valid API key or login session.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Tuple[User, UserAPIKey]:
    """
    Require API key authentication (strict API-only access).
    Use this for endpoints that should only be accessed via API.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    result = await APIKeyService.verify_api_key(credentials.credentials)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key, user = result
    return user, api_key


def check_permissions(required_permissions: list, user_permissions: dict = None, api_key_permissions: dict = None) -> bool:
    """Check if user has required permissions."""
    # For now, we'll use a simple permission check
    # This can be expanded later for more granular control
    
    if api_key_permissions:
        # Check API key permissions
        for perm in required_permissions:
            if not api_key_permissions.get(perm, False):
                return False
        return True
    
    # For session users, we'll check user role
    # This is a basic implementation - expand as needed
    return True  # Allow all authenticated users for now 