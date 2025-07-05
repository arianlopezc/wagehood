# Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability in Wagehood, please follow these steps:

1. **DO NOT** create a public GitHub issue for the vulnerability
2. Email security details to the project maintainers
3. Include steps to reproduce the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Best Practices

When using Wagehood:

### API Credentials
- **Never commit API keys** to the repository
- Use environment variables for all credentials
- Use `.env` files locally (never commit them)
- Rotate API keys regularly

### Configuration
- Review all configuration before deploying
- Use separate API keys for development/production
- Enable API key restrictions when possible
- Monitor API usage for anomalies

### Trading Safety
- Start with paper trading accounts
- Test strategies thoroughly before live trading
- Set appropriate position size limits
- Monitor automated trades closely
- Implement stop-loss mechanisms

## Secure Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/wagehood.git
cd wagehood

# Create .env file (do not commit)
cp .env.example .env

# Edit .env with your credentials
# NEVER share or commit this file
nano .env

# Install with security in mind
pip install -e . --user
```

## Credential Management

Wagehood uses encrypted credential storage for sensitive data:

```bash
# Configure credentials securely
wagehood config set-credential alpaca.api_key
# Enter key when prompted (not shown on screen)

# Credentials are encrypted and stored locally
# Never transmitted or logged
```

## Known Security Considerations

1. **Redis Security**: If using Redis, ensure it's properly secured:
   - Use authentication (`requirepass`)
   - Bind to localhost only
   - Use TLS for remote connections

2. **Log Files**: Wagehood sanitizes logs but always review before sharing:
   - API keys are masked
   - Sensitive data is redacted
   - Check logs before bug reports

3. **Network Security**: When running real-time services:
   - Use firewalls to restrict access
   - Monitor network connections
   - Use VPN for remote access

## Updates and Patches

- Watch the repository for security updates
- Update dependencies regularly
- Review changelogs for security fixes
- Test updates in development first

## Contact

For security concerns, contact the maintainers directly rather than using public issues.