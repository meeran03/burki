from typing import Optional, Dict, Any, List
import logging
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
import secrets
import re
from datetime import datetime

from app.db.database import get_db
from app.db.models import Organization, User, UserAPIKey
from app.services.billing_service import BillingService

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication and user management."""

    @staticmethod
    async def create_organization(
        name: str, 
        slug: Optional[str] = None, 
        description: Optional[str] = None,
        domain: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> Organization:
        """Create a new organization."""
        db = next(get_db())
        try:
            # Generate slug if not provided
            if not slug:
                slug = re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))
            
            # Check if slug already exists
            existing = db.query(Organization).filter(Organization.slug == slug).first()
            if existing:
                raise HTTPException(status_code=400, detail="Organization slug already exists")
            
            organization = Organization(
                name=name,
                slug=slug,
                description=description,
                domain=domain,
                settings=settings or {}
            )
            
            db.add(organization)
            db.commit()
            db.refresh(organization)
            
            logger.info(f"Created organization: {organization.name} (ID: {organization.id})")
            return organization
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error creating organization: {e}")
            raise HTTPException(status_code=400, detail="Organization with this slug already exists")
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating organization: {e}")
            raise HTTPException(status_code=500, detail="Failed to create organization")
        finally:
            db.close()

    @staticmethod
    async def get_organization_by_slug(slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        db = next(get_db())
        try:
            organization = db.query(Organization).filter(Organization.slug == slug).first()
            return organization
        finally:
            db.close()

    @staticmethod
    async def get_organization_by_id(organization_id: int) -> Optional[Organization]:
        """Get organization by ID."""
        db = next(get_db())
        try:
            organization = db.query(Organization).filter(Organization.id == organization_id).first()
            return organization
        finally:
            db.close()

    @staticmethod
    async def create_user(
        organization_id: int,
        email: str,
        first_name: str = "",
        last_name: str = "",
        full_name: Optional[str] = None,
        password: Optional[str] = None,
        google_id: Optional[str] = None,
        avatar_url: Optional[str] = None,
        role: str = "user",
        is_verified: bool = False
    ) -> User:
        """Create a new user."""
        db = next(get_db())
        try:
            # Check if user already exists in this organization
            existing = db.query(User).filter(
                User.organization_id == organization_id,
                User.email == email
            ).first()
            
            if existing:
                raise HTTPException(status_code=400, detail="User with this email already exists in organization")
            
            user = User(
                organization_id=organization_id,
                email=email,
                google_id=google_id,
                avatar_url=avatar_url,
                role=role,
                is_verified=is_verified
            )
            
            # Handle names
            if full_name:
                user.full_name = full_name
                user.split_full_name_if_needed()
            else:
                user.set_full_name(first_name, last_name)
            
            if password:
                user.set_password(password)
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Created user: {user.email} (ID: {user.id})")
            return user
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=400, detail="User with this email already exists")
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Failed to create user")
        finally:
            db.close()

    @staticmethod
    async def authenticate_user(email: str, password: str, organization_slug: str) -> Optional[User]:
        """Authenticate user with email and password."""
        db = next(get_db())
        try:
            # Get organization first
            organization = db.query(Organization).filter(Organization.slug == organization_slug).first()
            if not organization:
                return None
            
            # Get user with eager loading of organization
            user = db.query(User).options(
                # Eager load the organization relationship
                joinedload(User.organization)
            ).filter(
                User.organization_id == organization.id,
                User.email == email,
                User.is_active == True
            ).first()
            
            if not user or not user.check_password(password):
                return None
            
            # Update login tracking
            user.last_login_at = datetime.utcnow()
            user.login_count += 1
            db.commit()
            
            # Refresh to ensure all relationships are loaded
            db.refresh(user)
            
            logger.info(f"User authenticated: {user.email}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
        finally:
            db.close()

    @staticmethod
    async def authenticate_google_user(
        google_id: str, 
        email: str, 
        full_name: str, 
        avatar_url: str,
        organization_slug: str
    ) -> User:
        """Authenticate or create user from Google OAuth."""
        db = next(get_db())
        try:
            # Get organization
            organization = db.query(Organization).filter(Organization.slug == organization_slug).first()
            if not organization:
                raise HTTPException(status_code=404, detail="Organization not found")
            
            # Check if user exists by Google ID with eager loading
            user = db.query(User).options(joinedload(User.organization)).filter(
                User.google_id == google_id,
                User.organization_id == organization.id
            ).first()
            
            if user:
                # Update user info and login tracking
                user.full_name = full_name
                user.split_full_name_if_needed()
                user.avatar_url = avatar_url
                user.last_login_at = datetime.utcnow()
                user.login_count += 1
                user.is_verified = True  # Google users are auto-verified
            else:
                # Check if user exists by email with eager loading
                existing_user = db.query(User).options(joinedload(User.organization)).filter(
                    User.email == email,
                    User.organization_id == organization.id
                ).first()
                
                if existing_user:
                    # Link Google account to existing user
                    existing_user.google_id = google_id
                    existing_user.full_name = full_name
                    existing_user.split_full_name_if_needed()
                    existing_user.avatar_url = avatar_url
                    existing_user.last_login_at = datetime.utcnow()
                    existing_user.login_count += 1
                    existing_user.is_verified = True
                    user = existing_user
                else:
                    # Create new user
                    user = User(
                        organization_id=organization.id,
                        email=email,
                        full_name=full_name,
                        google_id=google_id,
                        avatar_url=avatar_url,
                        is_verified=True,
                        last_login_at=datetime.utcnow(),
                        login_count=1
                    )
                    user.split_full_name_if_needed()
                    db.add(user)
            
            db.commit()
            db.refresh(user)
            
            # Ensure organization is loaded
            if not hasattr(user, 'organization') or user.organization is None:
                user = db.query(User).options(joinedload(User.organization)).filter(User.id == user.id).first()
            
            logger.info(f"Google user authenticated: {user.email}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error authenticating Google user: {e}")
            raise HTTPException(status_code=500, detail="Failed to authenticate user")
        finally:
            db.close()

    @staticmethod
    async def get_user_by_id(user_id: int) -> Optional[User]:
        """Get user by ID."""
        db = next(get_db())
        try:
            user = db.query(User).options(joinedload(User.organization)).filter(User.id == user_id).first()
            return user
        finally:
            db.close()

    @staticmethod
    async def get_user_by_email(email: str, organization_id: int) -> Optional[User]:
        """Get user by email within organization."""
        db = next(get_db())
        try:
            user = db.query(User).filter(
                User.email == email,
                User.organization_id == organization_id
            ).first()
            return user
        finally:
            db.close()

    @staticmethod
    async def update_user(user_id: int, updates: Dict[str, Any]) -> Optional[User]:
        """Update user information."""
        db = next(get_db())
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            db.commit()
            db.refresh(user)
            
            logger.info(f"Updated user: {user.email}")
            return user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user: {e}")
            raise HTTPException(status_code=500, detail="Failed to update user")
        finally:
            db.close()


class APIKeyService:
    """Service for managing user API keys."""

    @staticmethod
    async def create_api_key(user_id: int, name: str) -> tuple[UserAPIKey, str]:
        """Create a new API key for user. Returns (api_key_record, plain_key)."""
        db = next(get_db())
        try:
            # Generate the key
            plain_key = UserAPIKey.generate_api_key()
            key_hash = UserAPIKey.hash_key(plain_key)
            key_prefix = UserAPIKey.get_key_prefix(plain_key)
            
            api_key = UserAPIKey(
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                key_prefix=key_prefix
            )
            
            db.add(api_key)
            db.commit()
            db.refresh(api_key)
            
            logger.info(f"Created API key '{name}' for user {user_id}")
            return api_key, plain_key
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to create API key")
        finally:
            db.close()

    @staticmethod
    async def get_user_api_keys(user_id: int) -> List[UserAPIKey]:
        """Get all API keys for a user."""
        db = next(get_db())
        try:
            api_keys = db.query(UserAPIKey).filter(
                UserAPIKey.user_id == user_id,
                UserAPIKey.is_active == True
            ).order_by(UserAPIKey.created_at.desc()).all()
            return api_keys
        finally:
            db.close()

    @staticmethod
    async def delete_api_key(user_id: int, api_key_id: int) -> bool:
        """Delete an API key."""
        db = next(get_db())
        try:
            api_key = db.query(UserAPIKey).filter(
                UserAPIKey.id == api_key_id,
                UserAPIKey.user_id == user_id
            ).first()
            
            if not api_key:
                return False
            
            db.delete(api_key)
            db.commit()
            
            logger.info(f"Deleted API key '{api_key.name}' for user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete API key")
        finally:
            db.close()

    @staticmethod
    async def verify_api_key(key: str) -> Optional[tuple[UserAPIKey, User]]:
        """Verify an API key and return the key record and user."""
        db = next(get_db())
        try:
            key_hash = UserAPIKey.hash_key(key)
            
            api_key = db.query(UserAPIKey).filter(
                UserAPIKey.key_hash == key_hash,
                UserAPIKey.is_active == True
            ).first()
            
            if not api_key:
                return None
            
            # Get the user
            user = db.query(User).filter(
                User.id == api_key.user_id,
                User.is_active == True
            ).first()
            
            if not user:
                return None
            
            # Update usage tracking
            api_key.last_used_at = datetime.utcnow()
            api_key.usage_count += 1
            db.commit()
            
            return api_key, user
            
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
        finally:
            db.close() 