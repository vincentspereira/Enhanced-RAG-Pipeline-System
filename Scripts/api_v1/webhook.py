import asyncio
import httpx
import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import logging
from .models import WebhookConfig, WebhookEvent

logger = logging.getLogger(__name__)

class WebhookManager:
    def __init__(self):
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._client = httpx.AsyncClient(timeout=10.0)
    
    def register_webhook(self, config: WebhookConfig) -> str:
        webhook_id = str(len(self._webhooks) + 1)
        config.created_at = datetime.utcnow()
        config.updated_at = config.created_at
        self._webhooks[webhook_id] = config
        return webhook_id
    
    def update_webhook(self, webhook_id: str, config: WebhookConfig) -> bool:
        if webhook_id not in self._webhooks:
            return False
        config.updated_at = datetime.utcnow()
        self._webhooks[webhook_id] = config
        return True
    
    def delete_webhook(self, webhook_id: str) -> bool:
        return bool(self._webhooks.pop(webhook_id, None))
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[Dict[str, WebhookConfig]]:
        return [{"id": k, "config": v} for k, v in self._webhooks.items()]
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def trigger_event(self, event: WebhookEvent):
        payload = event.dict()
        tasks = []
        
        for webhook_id, config in self._webhooks.items():
            if not config.is_active or event.event_type not in config.events:
                continue
            
            tasks.append(self._send_webhook(webhook_id, config, payload))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(self, webhook_id: str, config: WebhookConfig, payload: dict):
        try:
            headers = {"Content-Type": "application/json"}
            
            if config.secret:
                payload_str = json.dumps(payload)
                signature = self._generate_signature(payload_str, config.secret)
                headers["X-Webhook-Signature"] = signature
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    str(config.url),
                    json=payload,
                    headers=headers,
                    timeout=10.0
                )
                
                response.raise_for_status()
                logger.info(f"Successfully sent webhook {webhook_id} to {config.url}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook {webhook_id} to {config.url}: {str(e)}")
            # We might want to implement retry logic here

webhook_manager = WebhookManager()
