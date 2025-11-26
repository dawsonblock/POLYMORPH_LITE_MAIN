"""
Tests for input validation security utilities.

This module tests:
- Email validation
- Filename validation
- Path sanitization
- Password strength validation
- IP address validation
- Port validation
- XSS prevention
"""
import pytest
from fastapi import HTTPException
from retrofitkit.security.validators import InputValidator


class TestEmailValidation:
    """Test cases for email validation."""

    def test_valid_email(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.user@example.com",
            "user+tag@example.co.uk",
            "user_name@example-domain.com",
            "123@example.com",
        ]

        for email in valid_emails:
            result = InputValidator.validate_email(email)
            assert result == email.strip().lower()

    def test_email_normalized_to_lowercase(self):
        """Test that email is normalized to lowercase."""
        result = InputValidator.validate_email("User@EXAMPLE.COM")
        assert result == "user@example.com"

    def test_email_whitespace_stripped(self):
        """Test that whitespace is stripped from email."""
        result = InputValidator.validate_email("  user@example.com  ")
        assert result == "user@example.com"

    def test_invalid_email_format(self):
        """Test rejection of invalid email formats."""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user@@example.com",
            "user@example",
            "",
            "user @example.com",
            "user@exam ple.com",
        ]

        for email in invalid_emails:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_email(email)
            assert exc_info.value.status_code == 400
            assert "Invalid email format" in exc_info.value.detail

    def test_email_sql_injection_attempt(self):
        """Test that SQL injection attempts in email are rejected."""
        with pytest.raises(HTTPException):
            InputValidator.validate_email("admin'--@example.com")


class TestFilenameValidation:
    """Test cases for filename validation."""

    def test_valid_filename(self):
        """Test validation of safe filenames."""
        valid_filenames = [
            "document.pdf",
            "report_2024.xlsx",
            "data-file.csv",
            "image123.png",
            "file_name_123.txt",
        ]

        for filename in valid_filenames:
            result = InputValidator.validate_filename(filename)
            assert result == filename

    def test_filename_with_path_traversal(self):
        """Test rejection of filenames with path traversal attempts."""
        dangerous_filenames = [
            "../etc/passwd",
            "../../secret.txt",
            "file/../../../etc/passwd",
            "..\\windows\\system32",
        ]

        for filename in dangerous_filenames:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_filename(filename)
            assert exc_info.value.status_code == 400

    def test_filename_starting_with_dot(self):
        """Test rejection of hidden files (starting with dot)."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_filename(".hidden")
        assert exc_info.value.status_code == 400

    def test_filename_with_special_characters(self):
        """Test rejection of filenames with dangerous special characters."""
        dangerous_filenames = [
            "file;rm -rf /.txt",
            "file|cat /etc/passwd",
            "file&whoami.txt",
            "file$(cmd).txt",
            "file`ls`.txt",
        ]

        for filename in dangerous_filenames:
            with pytest.raises(HTTPException):
                InputValidator.validate_filename(filename)

    def test_empty_filename(self):
        """Test rejection of empty filename."""
        with pytest.raises(HTTPException):
            InputValidator.validate_filename("")


class TestPathSanitization:
    """Test cases for path sanitization."""

    def test_valid_path(self):
        """Test sanitization of valid paths."""
        valid_paths = [
            "data/files/document.txt",
            "reports/2024/report.pdf",
            "user_data/file.csv",
        ]

        for path in valid_paths:
            result = InputValidator.sanitize_path(path)
            assert not result.startswith('/')

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        dangerous_paths = [
            "../etc/passwd",
            "data/../../etc/passwd",
            "files/../../../secret",
            "..\\..\\windows\\system32",
        ]

        for path in dangerous_paths:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.sanitize_path(path)
            assert exc_info.value.status_code == 400
            assert "Invalid path" in exc_info.value.detail

    def test_path_leading_slash_removed(self):
        """Test that leading slashes are removed."""
        result = InputValidator.sanitize_path("/data/files/document.txt")
        assert not result.startswith('/')

    def test_path_with_special_characters(self):
        """Test rejection of paths with unsafe characters."""
        dangerous_paths = [
            "data;rm -rf /",
            "files|cat passwd",
            "data$HOME/file",
            "files`ls`",
        ]

        for path in dangerous_paths:
            with pytest.raises(HTTPException):
                InputValidator.sanitize_path(path)


class TestPasswordStrengthValidation:
    """Test cases for password strength validation."""

    def test_valid_strong_password(self):
        """Test validation of strong passwords."""
        strong_passwords = [
            "MyP@ssw0rd123!",
            "Str0ng!P@ssword",
            "C0mpl3x&Secure!",
            "Abcd1234!@#$%^&*()",
        ]

        for password in strong_passwords:
            result = InputValidator.validate_password_strength(password)
            assert result is True

    def test_password_too_short(self):
        """Test rejection of passwords under 12 characters."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_password_strength("Short1!")
        assert exc_info.value.status_code == 400
        assert "at least 12 characters" in exc_info.value.detail

    def test_password_missing_uppercase(self):
        """Test rejection of passwords without uppercase letters."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_password_strength("mypassword123!")
        assert exc_info.value.status_code == 400
        assert "uppercase letter" in exc_info.value.detail

    def test_password_missing_lowercase(self):
        """Test rejection of passwords without lowercase letters."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_password_strength("MYPASSWORD123!")
        assert exc_info.value.status_code == 400
        assert "lowercase letter" in exc_info.value.detail

    def test_password_missing_digit(self):
        """Test rejection of passwords without digits."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_password_strength("MyPassword!!!")
        assert exc_info.value.status_code == 400
        assert "digit" in exc_info.value.detail

    def test_password_missing_special_character(self):
        """Test rejection of passwords without special characters."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_password_strength("MyPassword123")
        assert exc_info.value.status_code == 400
        assert "special character" in exc_info.value.detail


class TestPortValidation:
    """Test cases for port number validation."""

    def test_valid_port_numbers(self):
        """Test validation of valid port numbers."""
        valid_ports = [1, 80, 443, 8080, 3000, 5432, 65535]

        for port in valid_ports:
            result = InputValidator.validate_port(port)
            assert result == port

    def test_port_below_range(self):
        """Test rejection of port numbers below valid range."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_port(0)
        assert exc_info.value.status_code == 400
        assert "Invalid port number" in exc_info.value.detail

    def test_port_above_range(self):
        """Test rejection of port numbers above valid range."""
        with pytest.raises(HTTPException) as exc_info:
            InputValidator.validate_port(65536)
        assert exc_info.value.status_code == 400

    def test_negative_port(self):
        """Test rejection of negative port numbers."""
        with pytest.raises(HTTPException):
            InputValidator.validate_port(-1)


class TestIPAddressValidation:
    """Test cases for IP address validation."""

    def test_valid_ip_addresses(self):
        """Test validation of valid IP addresses."""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
        ]

        for ip in valid_ips:
            result = InputValidator.validate_ip_address(ip)
            assert result == ip

    def test_invalid_ip_format(self):
        """Test rejection of invalid IP formats."""
        invalid_ips = [
            "256.1.1.1",          # Out of range
            "192.168.1",          # Missing octet
            "192.168.1.1.1",      # Too many octets
            "192.168.-1.1",       # Negative number
            "192.168.1.abc",      # Non-numeric
            "not.an.ip.address",  # Non-numeric
            "",                   # Empty
        ]

        for ip in invalid_ips:
            with pytest.raises(HTTPException) as exc_info:
                InputValidator.validate_ip_address(ip)
            assert exc_info.value.status_code == 400
            assert "Invalid IP address" in exc_info.value.detail


class TestStringSanitization:
    """Test cases for string sanitization (XSS prevention)."""

    def test_clean_string_unchanged(self):
        """Test that clean strings pass through unchanged."""
        clean_text = "This is a normal string with 123 numbers."
        result = InputValidator.sanitize_string(clean_text)
        assert result == clean_text

    def test_html_tags_removed(self):
        """Test that HTML tags are removed."""
        dangerous_strings = [
            ("<script>alert('xss')</script>", "alert('xss')"),
            ("Hello <b>World</b>", "Hello World"),
            ("<img src=x onerror=alert(1)>", ""),
            ("Text<script>evil()</script>More", "TextMore"),
        ]

        for dangerous, expected in dangerous_strings:
            result = InputValidator.sanitize_string(dangerous)
            assert result == expected

    def test_string_length_limited(self):
        """Test that long strings are truncated."""
        long_string = "A" * 2000
        result = InputValidator.sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_empty_string_handled(self):
        """Test that empty string is handled correctly."""
        result = InputValidator.sanitize_string("")
        assert result == ""

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        result = InputValidator.sanitize_string("  text  ")
        assert result == "text"

    def test_nested_html_tags_removed(self):
        """Test that nested HTML tags are removed."""
        dangerous = "<div><script>alert(1)</script><p>text</p></div>"
        result = InputValidator.sanitize_string(dangerous)
        assert "<" not in result
        assert ">" not in result

    def test_xss_event_handlers_removed(self):
        """Test that XSS event handlers are removed."""
        dangerous_inputs = [
            "<img onerror='alert(1)' src=x>",
            "<body onload='alert(1)'>",
            "<div onclick='steal()'>Click</div>",
        ]

        for dangerous in dangerous_inputs:
            result = InputValidator.sanitize_string(dangerous)
            assert "onerror" not in result.lower()
            assert "onload" not in result.lower()
            assert "onclick" not in result.lower()


class TestSecurityEdgeCases:
    """Test edge cases and corner cases for security validators."""

    def test_null_byte_injection(self):
        """Test handling of null byte injection attempts."""
        # Null bytes can be used to bypass filters
        dangerous_filename = "file.txt\x00.exe"
        with pytest.raises(HTTPException):
            InputValidator.validate_filename(dangerous_filename)

    def test_unicode_normalization(self):
        """Test handling of Unicode normalization attacks."""
        # Some Unicode characters can look like path separators
        result = InputValidator.sanitize_string("file\u2044etc\u2044passwd")
        # Should not contain actual path separators
        assert result == "file\u2044etc\u2044passwd"

    def test_extremely_long_input(self):
        """Test handling of extremely long inputs."""
        very_long = "A" * 1_000_000
        result = InputValidator.sanitize_string(very_long, max_length=1000)
        assert len(result) == 1000

    def test_email_with_quotes(self):
        """Test email validation with quoted strings."""
        # These are technically valid but often indicate injection attempts
        with pytest.raises(HTTPException):
            InputValidator.validate_email('"user"@example.com')
