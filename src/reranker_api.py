"""Remote reranker API clients used by offline experiments."""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Dict, List, Optional

import requests


SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
SILICONFLOW_QWEN3_RERANKER_MODELS = [
  "Qwen/Qwen3-Reranker-0.6B",
  "Qwen/Qwen3-Reranker-4B",
  "Qwen/Qwen3-Reranker-8B",
]
SILICONFLOW_CHUNK_OPTION_MODELS = {
  "BAAI/bge-reranker-v2-m3",
  "Pro/BAAI/bge-reranker-v2-m3",
  "netease-youdao/bce-reranker-base_v1",
}
SILICONFLOW_QWEN3_PRICE_PER_M_TOKEN = {
  "Qwen/Qwen3-Reranker-0.6B": 0.01,
  "Qwen/Qwen3-Reranker-4B": 0.02,
  "Qwen/Qwen3-Reranker-8B": 0.04,
}
DEFAULT_QWEN3_RERANK_INSTRUCTION = (
  "Given an academic paper recommendation query, rerank candidate papers by how well "
  "their titles and abstracts satisfy the user's research interest."
)


def _percentile(values: List[float], percentile: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * percentile
  low = int(rank)
  high = min(low + 1, len(ordered) - 1)
  weight = rank - low
  return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _extract_tokens(payload: Dict[str, Any]) -> Dict[str, int]:
  tokens: Dict[str, Any] = {}
  if isinstance(payload.get("tokens"), dict):
    tokens = payload.get("tokens") or {}
  elif isinstance(payload.get("meta"), dict):
    meta = payload.get("meta") or {}
    if isinstance(meta.get("tokens"), dict):
      tokens = meta.get("tokens") or {}
  elif isinstance(payload.get("meta"), list):
    for item in payload.get("meta") or []:
      if isinstance(item, dict) and isinstance(item.get("tokens"), dict):
        tokens = item.get("tokens") or {}
        break

  def _as_int(value: Any) -> int:
    try:
      return max(int(value or 0), 0)
    except Exception:
      return 0

  return {
    "input_tokens": _as_int(tokens.get("input_tokens")),
    "output_tokens": _as_int(tokens.get("output_tokens")),
  }


class SiliconFlowReranker:
  """Small adapter matching src/3.rank_papers.py's reranker interface."""

  def __init__(
    self,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
    instruction: Optional[str] = DEFAULT_QWEN3_RERANK_INSTRUCTION,
    return_documents: bool = False,
    max_chunks_per_doc: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
    session: Optional[requests.Session] = None,
  ) -> None:
    self.api_key = (
      api_key
      or os.getenv("SILICONFLOW_API_KEY")
      or os.getenv("RERANK_API_KEY")
      or ""
    ).strip()
    if not self.api_key:
      raise RuntimeError("missing SILICONFLOW_API_KEY or RERANK_API_KEY")

    self.base_url = (
      base_url
      or os.getenv("SILICONFLOW_RERANK_URL")
      or os.getenv("RERANK_API_BASE_URL")
      or SILICONFLOW_RERANK_URL
    ).strip()
    self.timeout = max(int(timeout or 1), 1)
    self.instruction = str(instruction or "").strip()
    self.return_documents = bool(return_documents)
    self.max_chunks_per_doc = max_chunks_per_doc
    self.overlap_tokens = overlap_tokens
    self.session = session or requests.Session()
    self.call_count = 0
    self.total_latency_seconds = 0.0
    self.latencies_seconds: List[float] = []
    self.input_tokens = 0
    self.output_tokens = 0

  def rerank(
    self,
    *,
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    model: Optional[str] = None,
  ) -> Dict[str, Any]:
    query_text = str(query or "").strip()
    docs = [str(doc or "") for doc in documents]
    if not query_text:
      raise ValueError("rerank query 不能为空")
    if not docs:
      raise ValueError("rerank documents 不能为空")

    payload: Dict[str, Any] = {
      "model": str(model or SILICONFLOW_QWEN3_RERANKER_MODELS[0]),
      "query": query_text,
      "documents": docs,
      "return_documents": self.return_documents,
    }
    if top_n is not None:
      payload["top_n"] = max(int(top_n), 1)
    if self.instruction and self._supports_instruction(payload["model"]):
      payload["instruction"] = self.instruction
    if (
      self.max_chunks_per_doc is not None
      and self._supports_chunk_options(payload["model"])
    ):
      payload["max_chunks_per_doc"] = max(int(self.max_chunks_per_doc), 1)
    if (
      self.overlap_tokens is not None
      and self._supports_chunk_options(payload["model"])
    ):
      payload["overlap_tokens"] = min(max(int(self.overlap_tokens), 0), 80)

    started = time.perf_counter()
    response = self.session.post(
      self.base_url,
      headers={
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
      },
      json=payload,
      timeout=self.timeout,
    )
    elapsed = time.perf_counter() - started
    self.call_count += 1
    self.total_latency_seconds += elapsed
    self.latencies_seconds.append(elapsed)

    try:
      response.raise_for_status()
    except requests.HTTPError as exc:
      text = getattr(response, "text", "") or ""
      raise requests.HTTPError(
        f"SiliconFlow rerank API failed: status={response.status_code} body={text[:500]}"
      ) from exc

    data = response.json()
    if not isinstance(data, dict):
      raise RuntimeError("SiliconFlow rerank API response is not a JSON object")
    if "results" not in data:
      raise RuntimeError("SiliconFlow rerank API response missing results")

    token_usage = _extract_tokens(data)
    self.input_tokens += token_usage["input_tokens"]
    self.output_tokens += token_usage["output_tokens"]
    return data

  @staticmethod
  def _supports_instruction(model: str) -> bool:
    return str(model or "").strip() in SILICONFLOW_QWEN3_RERANKER_MODELS

  @staticmethod
  def _supports_chunk_options(model: str) -> bool:
    return str(model or "").strip() in SILICONFLOW_CHUNK_OPTION_MODELS

  def stats(self, model: str = "") -> Dict[str, Any]:
    price = SILICONFLOW_QWEN3_PRICE_PER_M_TOKEN.get(str(model or ""))
    estimated_cost = None
    if price is not None:
      estimated_cost = round((self.input_tokens + self.output_tokens) * price / 1_000_000, 8)
    mean_latency = (
      statistics.fmean(self.latencies_seconds)
      if self.latencies_seconds
      else 0.0
    )
    return {
      "api_calls": self.call_count,
      "latency_seconds_total": round(self.total_latency_seconds, 3),
      "latency_seconds_mean": round(mean_latency, 3),
      "latency_seconds_p95": round(_percentile(self.latencies_seconds, 0.95), 3),
      "input_tokens": self.input_tokens,
      "output_tokens": self.output_tokens,
      "estimated_cost_usd": estimated_cost,
      "price_per_m_token_usd": price,
    }
