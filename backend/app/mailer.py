"""Outbound email over SMTP, with an explicit "not configured" state.

The project has no email vendor and does not need one: SMTP is the common denominator every
provider speaks (Gmail app passwords, Resend, Postmark, Mailgun, a company relay). Nothing here runs
unless ``SMTP_HOST`` and ``SMTP_FROM`` are set, and a deployment that never configures them still
starts, signs users up and serves recommendations — it just cannot send a reset link, which the API
reports honestly instead of pretending the mail went out.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, parseaddr

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class EmailNotConfigured(RuntimeError):
    """No SMTP settings, so no message can be sent from this instance."""


class EmailDeliveryError(RuntimeError):
    """The server was configured but the message could not be handed over."""


@dataclass(frozen=True)
class Email:
    to: str
    subject: str
    text: str


def send_email(settings: Settings, email: Email) -> None:
    """Deliver one plain-text message.

    Raises ``EmailNotConfigured`` before touching the network when settings are missing, and
    ``EmailDeliveryError`` when the handover fails, so the caller can decide what the user is told.
    """
    if not settings.smtp_configured:
        raise EmailNotConfigured("SMTP is not configured")

    message = EmailMessage()
    display_name, address = parseaddr(settings.smtp_from)
    message["From"] = formataddr((display_name, address)) if display_name else address
    message["To"] = email.to
    message["Subject"] = email.subject
    message.set_content(email.text)

    try:
        if settings.smtp_ssl:
            server: smtplib.SMTP = smtplib.SMTP_SSL(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
            )
        else:
            server = smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=settings.smtp_timeout_seconds,
            )
        with server:
            if settings.smtp_starttls and not settings.smtp_ssl:
                server.starttls()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(message, from_addr=address, to_addrs=[email.to])
    except (OSError, smtplib.SMTPException) as exc:
        logger.warning("Could not send mail to %s", email.to, exc_info=True)
        raise EmailDeliveryError(str(exc)) from exc


def reset_email(to: str, *, link: str, ttl_minutes: int, app_name: str) -> Email:
    """The reset message: one link, the expiry, and no request to reply to it."""
    return Email(
        to=to,
        subject=f"Reset your {app_name} password",
        text=(
            "Someone asked to reset the password for this account.\n\n"
            f"Open this link to choose a new one:\n\n{link}\n\n"
            f"The link works once and expires in {ttl_minutes} minutes. "
            "If you did not ask for this, ignore the message — your password is unchanged "
            "and the link expires on its own.\n"
        ),
    )
