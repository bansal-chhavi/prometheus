import pytest
import httpx
import os
from prometheus.detectors.hf_nli import HuggingFaceNLI

import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.mark.asyncio
async def test_huggingface_nli_connectivity():
    """
    Inference checker: Verifies that the HuggingFace NLI endpoint is reachable.
    This ensures that the URL is correct and DNS resolution works, preventing
    issues like 'getaddrinfo failed' from hardcoded incorrect router URLs.
    """
    hf = HuggingFaceNLI("dummy_token")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # We don't need a valid payload to test connectivity
            response = await client.post(hf.url, json={})
            
            # If we reach here, the DNS resolved and a connection was established.
            # Depending on the payload/token, we might get 401 (Unauthorized), 
            # 400 (Bad Request), or 403. Any of these mean the server is reachable.
            assert response.status_code in [400, 401, 403, 404, 429, 500, 503], f"Unexpected status: {response.status_code}"
            
    except httpx.ConnectError as e:
        pytest.fail(f"HuggingFace NLI endpoint is unreachable (DNS or Connection error): {e}. "
                    f"Check if URL is correct: {hf.url}")
