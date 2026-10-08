import json

import numpy as np
import pytest
from botocore.exceptions import ClientError

from fakes import FakeBedrock
from simtest.embedder import EmbeddingCache, HashEmbedder, TitanEmbedder


def titan(client, **kw):
    kw.setdefault("sleep", lambda s: None)
    return TitanEmbedder(client=client, cache=EmbeddingCache(None), **kw)


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder(dim=128)
    a, b = e.embed("payment gateway times out"), e.embed("payment gateway times out", use_cache=False)
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_hash_embedder_similar_text_is_closer():
    e = HashEmbedder(dim=256)
    base = e.embed("billing portal login page times out with error ERR-1234")
    near = e.embed("billing portal login page times out error ERR-1234 in prod")
    far = e.embed("mobile app csv importer duplicates records ERR-9999")
    assert float(base @ near) > float(base @ far) + 0.2


def test_titan_v2_request_body_has_dimensions_and_normalize():
    fake = FakeBedrock()
    e = titan(fake, dimensions=512)
    vec = e.embed("hello world")
    body = fake.calls[0]["body"]
    assert fake.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert body == {"inputText": "hello world", "dimensions": 512, "normalize": True}
    assert vec.shape == (512,)


def test_titan_v1_omits_dimensions_and_is_1536():
    fake = FakeBedrock()
    e = titan(fake, model_id="amazon.titan-embed-text-v1")
    assert e.dim == 1536
    e.embed("hello")
    assert fake.calls[0]["body"] == {"inputText": "hello"}


def test_titan_retries_throttling_then_succeeds():
    fake = FakeBedrock(fail_first=3)
    e = titan(fake, dimensions=256)
    e.embed("retry me")
    assert len(fake.calls) == 4
    assert e.stats["throttles"] == 3 and e.stats["retries"] == 3


def test_titan_gives_up_after_max_retries():
    fake = FakeBedrock(fail_first=100)
    e = titan(fake, dimensions=256, max_retries=2)
    with pytest.raises(RuntimeError, match="after 2 retries"):
        e.embed("never works")
    assert len(fake.calls) == 3


def test_titan_non_retryable_error_raises_immediately():
    fake = FakeBedrock(fail_first=1, code="ValidationException")
    e = titan(fake, dimensions=256)
    with pytest.raises(ClientError):
        e.embed("bad request")
    assert len(fake.calls) == 1


def test_titan_validates_dimensions_and_empty_text():
    with pytest.raises(ValueError):
        titan(FakeBedrock(), dimensions=999)
    with pytest.raises(ValueError):
        titan(FakeBedrock(), dimensions=256).embed("   ")


def test_titan_truncates_long_text():
    fake = FakeBedrock()
    e = titan(fake, dimensions=256, max_chars=50)
    e.embed("x" * 500)
    assert len(fake.calls[0]["body"]["inputText"]) == 50
    assert e.stats["truncated"] == 1


def test_cache_prevents_repeat_calls_and_embed_many_dedupes():
    fake = FakeBedrock()
    e = titan(fake, dimensions=256)
    e.embed("same text")
    e.embed("same text")
    assert len(fake.calls) == 1 and e.stats["cache_hits"] == 1
    out = e.embed_many(["a", "b", "a", "same text", "b"], workers=4)
    assert out.shape == (5, 256)
    assert np.allclose(out[0], out[2]) and np.allclose(out[1], out[4])
    assert len(fake.calls) == 3  # same text was cached; a and b embedded once each
