# Authentication System Setup

This document outlines the complete authentication system implemented for Burqi Voice AI.

## Overview

The authentication system includes:
- **Organizations**: Multi-tenant support where each organization has its own data
- **Users**: Users belong to organizations and can have different roles
- **Google OAuth**: Sign in with Google integration
- **Manual Authentication**: Email/password authentication with secure password hashing
- **API Keys**: User-generated API keys for programmatic access
- **Session Management**: Secure session-based authentication for web interface

## Database Models

### Organization
- `id`: Primary key
- `name`: Organization display name
- `slug`: Unique URL-friendly identifier
- `description`: Optional description
- `domain`: Optional domain for email-based signup restrictions
- `settings`: JSON field for organization-specific settings
- `is_active`: Whether the organization is active

### User
- `id`: Primary key
- `organization_id`: Foreign key to Organization
- `email`: User's email address (unique within organization)
- `full_name`: User's display name
- `password_hash`: Bcrypt-hashed password (nullable for OAuth-only users)
- `google_id`: Google OAuth user ID (nullable)
- `avatar_url`: Profile picture URL
- `is_active`: Whether the user account is active
- `is_verified`: Whether the user's email is verified
- `role`: User role (admin, user, viewer)
- `last_login_at`: Last login timestamp
- `login_count`: Number of times user has logged in
- `preferences`: JSON field for user preferences

### UserAPIKey
- `id`: Primary key
- `user_id`: Foreign key to User
- `name`: Descriptive name for the API key
- `key_hash`: SHA256 hash of the actual key
- `key_prefix`: First 12 characters + "..." for identification
- `last_used_at`: When the key was last used
- `usage_count`: Number of times the key has been used
- `is_active`: Whether the key is active
- `permissions`: JSON field defining access permissions
- `rate_limit`: JSON field for rate limiting settings

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Security Keys (REQUIRED)
SECRET_KEY=your-secret-key-for-sessions-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-this-in-production

# Google OAuth (REQUIRED for Google Sign-In)
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret

# Database (REQUIRED)
DATABASE_URL=postgresql://username:password@localhost:5432/diwaar
```

## Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
5. Set application type to "Web application"
6. Add authorized redirect URIs:
   - `http://localhost:8000` (development)
   - `https://yourdomain.com` (production)
7. Copy the Client ID and Client Secret to your `.env` file

## API Endpoints

### Authentication Routes
- `GET /auth/login` - Login page
- `POST /auth/login` - Handle manual login
- `POST /auth/google` - Handle Google OAuth login
- `GET /auth/register` - Registration page
- `POST /auth/register` - Handle manual registration
- `POST /auth/google-register` - Handle Google OAuth registration
- `GET /auth/logout` - Logout user

### API Key Management
- `GET /auth/api-keys` - API keys management page
- `POST /auth/api-keys/create` - Create new API key
- `POST /auth/api-keys/{key_id}/delete` - Delete API key

## API Key Usage

API keys can be used to authenticate API requests:

```bash
# Include in Authorization header
curl -H "Authorization: Bearer diwaar_your_api_key_here" \
     https://yourdomain.com/api/v1/assistants
```

## Features

### Multi-Tenant Architecture
- Each organization has isolated data
- Users can only access their organization's assistants and calls
- Cross-organization data access is prevented

### Role-Based Access Control
- **Admin**: Full access including user management
- **User**: Standard access to create/manage assistants and view calls
- **Viewer**: Read-only access

### Security Features
- Bcrypt password hashing
- Secure session management
- CSRF protection via session tokens
- API key rate limiting
- Organization-based data isolation

### User Experience
- Modern responsive UI with Bootstrap 5
- Google Sign-In integration
- Password strength validation
- Real-time form validation
- API key management with one-time display

## Database Migration

The authentication tables are already migrated. If you need to run migrations:

```bash
# Run pending migrations
alembic upgrade head

# Create new migration (if needed)
alembic revision --autogenerate -m "Description of changes"
```

## Security Considerations

### Production Deployment
1. **Use strong secret keys**: Generate random 32+ character strings
2. **Enable HTTPS**: All authentication should happen over SSL
3. **Configure CORS**: Restrict origins to your actual domains
4. **Rate limiting**: Implement API rate limiting in production
5. **Monitor API keys**: Track usage and detect anomalies

### Password Policy
- Minimum 8 characters
- Must include uppercase, lowercase, number, and special character
- Passwords are hashed with bcrypt

### Session Security
- Sessions expire automatically
- Secure session cookies
- Session data is server-side only

## Testing

### Manual Testing
1. Visit `/auth/register` to create a new organization and user
2. Test login with email/password
3. Test Google OAuth login
4. Create API keys in `/auth/api-keys`
5. Test API key authentication

### Required Test Data
- Valid Google OAuth credentials
- Test email accounts
- Strong passwords meeting policy requirements

## Troubleshooting

### Common Issues

1. **Google OAuth not working**
   - Check GOOGLE_CLIENT_ID is correctly set
   - Verify redirect URIs in Google Console
   - Ensure Google+ API is enabled

2. **Session not persisting**
   - Check SECRET_KEY is set
   - Verify session middleware is configured
   - Check browser cookies are enabled

3. **Database connection errors**
   - Verify DATABASE_URL format
   - Check database server is running
   - Run migrations with `alembic upgrade head`

4. **API key authentication failing**
   - Verify key format: `diwaar_` + 32 random characters
   - Check Authorization header format
   - Ensure API key is active

## Next Steps

After completing authentication setup:

1. **Update existing routes**: Ensure all routes require authentication
2. **Add admin panel**: For organization and user management
3. **Implement email verification**: For enhanced security
4. **Add audit logging**: Track user actions and API usage
5. **Set up monitoring**: Monitor failed login attempts and API usage
6. **Configure backup**: Regular database backups including user data 