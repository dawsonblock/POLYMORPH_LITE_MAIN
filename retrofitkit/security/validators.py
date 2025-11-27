"""
Input validation utilities for API security.
"""
import re
from fastapi import HTTPException


class InputValidator:
    """Validates and sanitizes user inputs."""

    # Regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    PATH_SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-/\.]+$')

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        if not email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        email = email.strip().lower()
        if not InputValidator.EMAIL_PATTERN.match(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        return email

    @staticmethod
    def validate_filename(filename: str) -> str:
        """
        Validate filename to prevent path traversal.
        Only allows alphanumeric, underscore, dash, and dot.
        """
        if not filename or not InputValidator.FILENAME_PATTERN.match(filename):
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Additional checks
        if filename.startswith('.') or '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        return filename

    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """
        Sanitize string input to prevent XSS.
        Removes HTML tags and limits length.
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Limit length
        if len(text) > max_length:
            text = text[:max_length]

        return text.strip()

    @staticmethod
    def validate_password_strength(password: str) -> bool:
        """
        Validate password meets security requirements.
        
        Requirements:
        - At least 12 characters
        - Contains uppercase letter
        - Contains lowercase letter
        - Contains digit
        - Contains special character
        """
        if len(password) < 12:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 12 characters"
            )

        if not re.search(r'[A-Z]', password):
            raise HTTPException(
                status_code=400,
                detail="Password must contain uppercase letter"
            )

        if not re.search(r'[a-z]', password):
            raise HTTPException(
                status_code=400,
                detail="Password must contain lowercase letter"
            )

        if not re.search(r'\d', password):
            raise HTTPException(
                status_code=400,
                detail="Password must contain digit"
            )

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise HTTPException(
                status_code=400,
                detail="Password must contain special character"
            )

        return True

    @staticmethod
    def validate_port(port: int) -> int:
        """Validate port number is in valid range."""
        if not (1 <= port <= 65535):
            raise HTTPException(status_code=400, detail="Invalid port number")
        return port

    @staticmethod
    def validate_ip_address(ip: str) -> str:
        """Basic IP address validation."""
        parts = ip.split('.')
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="Invalid IP address")

        try:
            for part in parts:
                num = int(part)
                if not (0 <= num <= 255):
                    raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid IP address")

        return ip

    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize file path to prevent directory traversal.
        """
        # Remove any .. sequences
        if '..' in path:
            raise HTTPException(status_code=400, detail="Invalid path")

        # Remove leading slashes
        path = path.lstrip('/')

        # Only allow safe characters
        if not InputValidator.PATH_SAFE_PATTERN.match(path):
            raise HTTPException(status_code=400, detail="Invalid path characters")

        return path
