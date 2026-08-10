#!/usr/bin/env python3
"""Run a resumable MMLU-Pro score/parity evaluation through OpenAI HTTP APIs.

The runner intentionally depends only on Python's standard library.  It pins the
official dataset revision, preserves the official five-shot grouping, appends one
durable JSONL record per endpoint/question, and can resume after interruption.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
OFFICIAL_HARNESS_REVISION = "f418b116db00b065c2aea046518d8fcf74d39872"
EXPECTED_SPLIT_COUNTS = {"validation": 70, "test": 12032}
CHOICES = "ABCDEFGHIJ"
PROMPT_VERSION = "mmlu-pro-official-five-shot-v1"
DIRECT_PROMPT_VERSION = "mmlu-pro-five-shot-direct-system-letter-v2"
DIRECT_SYSTEM_PROMPT = (
    "You are a multiple-choice grading engine. Reply with exactly one uppercase "
    "ASCII letter from A through J. Do not explain, reason, or emit any other text."
)
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"

ANSWER_PATTERNS = (
    re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"answer:\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\b([A-J])\b"),
)


class EvalError(RuntimeError):
    """A benchmark contract or HTTP request failed."""


class HttpStatusError(EvalError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:1000]}")
        self.status = status
        self.body = body
        self.url = url


@dataclasses.dataclass(frozen=True)
class Endpoint:
    side: str
    base_url: str
    model: str
    api_key: Optional[str]

    @property
    def completion_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    @property
    def models_url(self) -> str:
        completion = self.completion_url
        return completion[: -len("/chat/completions")] + "/models"

    @property
    def endpoint_id(self) -> str:
        material = f"{self.side}\0{self.base_url.rstrip('/')}\0{self.model}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @property
    def public_identity(self) -> dict[str, str]:
        return {
            "side": self.side,
            "base_url": self.base_url.rstrip("/"),
            "model": self.model,
            "endpoint_id": self.endpoint_id,
        }


@dataclasses.dataclass(frozen=True)
class Task:
    sequence: int
    question_id: int
    category: str
    source: str
    answer: str
    prompt: str
    prompt_sha256: str
    shared_prefix_sha256: str


class ResultWriter:
    def __init__(self, path: Path, fsync_every: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.fsync_every = max(1, fsync_every)
        self.since_fsync = 0
        self.lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.lock:
            self.handle.write(line + "\n")
            self.handle.flush()
            self.since_fsync += 1
            if self.since_fsync >= self.fsync_every:
                os.fsync(self.handle.fileno())
                self.since_fsync = 0

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_runtime_evidence(paths: Iterable[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            payload_bytes = resolved.read_bytes()
        except OSError as error:
            raise EvalError(f"could not read runtime evidence {resolved}: {error}") from error
        try:
            payload = json.loads(payload_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvalError(f"runtime evidence {resolved} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise EvalError(f"runtime evidence {resolved} must contain a JSON object")
        evidence.append(
            {
                "path": str(resolved),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "payload": payload,
            }
        )
    return evidence


def load_question_id_files(
    paths: Iterable[Path],
) -> tuple[list[int], list[dict[str, Any]]]:
    question_ids: list[int] = []
    evidence: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in paths:
        resolved = path.resolve()
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            raise EvalError(
                f"could not read question-ID file {resolved}: {error}"
            ) from error
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise EvalError(f"question-ID file {resolved} is not UTF-8") from error
        file_count = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            value = line.strip()
            if not value:
                continue
            if re.fullmatch(r"[0-9]+", value) is None:
                raise EvalError(
                    f"question-ID file {resolved}:{line_number} is not a "
                    "non-negative decimal integer"
                )
            question_id = int(value)
            if question_id in seen:
                raise EvalError(
                    f"duplicate question ID {question_id} in question-ID files"
                )
            seen.add(question_id)
            question_ids.append(question_id)
            file_count += 1
        if file_count == 0:
            raise EvalError(f"question-ID file {resolved} contains no IDs")
        evidence.append(
            {
                "path": str(resolved),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "question_ids": file_count,
            }
        )
    return question_ids, evidence


def open_json_url(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
    direct: bool = False,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    data = None
    if payload is not None:
        data = canonical_json(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if direct
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        body = raw.decode("utf-8", errors="replace")
        raise HttpStatusError(error.code, body, url) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise EvalError(f"request to {url} failed: {error}") from error
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvalError(f"request to {url} returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise EvalError(f"request to {url} returned a non-object JSON response")
    return decoded


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    required = {
        "question_id",
        "question",
        "options",
        "answer",
        "answer_index",
        "cot_content",
        "category",
        "src",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise EvalError(f"dataset row is missing fields: {missing}")
    options = [str(option) for option in raw["options"] if option != "N/A"]
    answer = str(raw["answer"]).upper()
    answer_index = int(raw["answer_index"])
    if not 2 <= len(options) <= len(CHOICES):
        raise EvalError(f"question {raw['question_id']} has {len(options)} usable options")
    if answer not in CHOICES[: len(options)]:
        raise EvalError(f"question {raw['question_id']} has invalid answer {answer!r}")
    if answer_index != CHOICES.index(answer):
        raise EvalError(
            f"question {raw['question_id']} answer/index mismatch: "
            f"{answer!r}/{answer_index}"
        )
    return {
        "question_id": int(raw["question_id"]),
        "question": str(raw["question"]),
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "cot_content": str(raw["cot_content"] or ""),
        "category": str(raw["category"]),
        "src": str(raw["src"]),
    }


def validate_split(rows: list[dict[str, Any]], split: str, revision: str) -> None:
    expected = EXPECTED_SPLIT_COUNTS.get(split)
    if revision == DATASET_REVISION and expected is not None and len(rows) != expected:
        raise EvalError(
            f"pinned {split} split contains {len(rows)} rows; expected {expected}"
        )
    question_ids = [row["question_id"] for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise EvalError(f"{split} split contains duplicate question IDs")
    if not rows:
        raise EvalError(f"{split} split is empty")


def fetch_dataset_split(
    cache_dir: Path,
    split: str,
    revision: str,
    *,
    page_delay_seconds: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"mmlu-pro-{revision[:12]}-{split}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if not isinstance(cached, dict) or cached.get("revision") != revision:
            raise EvalError(f"dataset cache metadata mismatch in {cache_path}")
        rows = [normalize_row(row) for row in cached.get("rows", [])]
        validate_split(rows, split, revision)
        return rows

    page_cache_dir = cache_dir / f"mmlu-pro-{revision[:12]}-{split}-pages"
    page_cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None
    while total is None or offset < total:
        page_cache_path = page_cache_dir / f"offset-{offset:08d}.json"
        fetched = False
        if page_cache_path.exists():
            with page_cache_path.open("r", encoding="utf-8") as handle:
                page_document = json.load(handle)
            if (
                not isinstance(page_document, dict)
                or page_document.get("revision") != revision
                or page_document.get("split") != split
                or page_document.get("offset") != offset
            ):
                raise EvalError(f"dataset page cache mismatch in {page_cache_path}")
            page_rows = page_document.get("rows")
            total = int(page_document.get("num_rows_total", -1))
        else:
            query = urllib.parse.urlencode(
                {
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": split,
                    "offset": offset,
                    "length": 100,
                    "revision": revision,
                }
            )
            page_url = f"{DATASET_ROWS_URL}?{query}"
            page = None
            for attempt in range(9):
                try:
                    page = open_json_url(page_url, timeout=60.0)
                    break
                except HttpStatusError as error:
                    if error.status != 429 or attempt == 8:
                        raise
                    wait_seconds = min(60.0, 5.0 * (2**attempt))
                    print(
                        f"dataset {split}: rate limited at offset {offset}; "
                        f"retrying in {wait_seconds:.0f}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                except EvalError:
                    if attempt == 8:
                        raise
                    wait_seconds = min(30.0, 2.0 * (2**attempt))
                    print(
                        f"dataset {split}: transport retry at offset {offset} "
                        f"in {wait_seconds:.0f}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
            if page is None:
                raise EvalError(f"dataset page at offset {offset} exhausted retries")
            page_rows = page.get("rows")
            total = int(page.get("num_rows_total", -1))
            page_document = {
                "dataset": DATASET_ID,
                "revision": revision,
                "split": split,
                "offset": offset,
                "num_rows_total": total,
                "rows": page_rows,
            }
            atomic_write_json(page_cache_path, page_document)
            fetched = True
        if not isinstance(page_rows, list):
            raise EvalError(f"dataset server returned no rows for {split} offset {offset}")
        assert total is not None
        if total < 0:
            raise EvalError("dataset server omitted num_rows_total")
        if not page_rows and offset < total:
            raise EvalError(f"dataset server returned an empty page at offset {offset}")
        for item in page_rows:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                raise EvalError(f"dataset server returned a malformed row at offset {offset}")
            truncated = item.get("truncated_cells") or []
            if truncated:
                raise EvalError(
                    f"dataset server truncated cells for {split} row {item.get('row_idx')}"
                )
            rows.append(normalize_row(item["row"]))
        offset += len(page_rows)
        print(f"dataset {split}: {offset}/{total}", flush=True)
        if fetched and offset < total and page_delay_seconds > 0:
            time.sleep(page_delay_seconds)

    validate_split(rows, split, revision)
    document = {
        "dataset": DATASET_ID,
        "revision": revision,
        "split": split,
        "row_count": len(rows),
        "rows_sha256": sha256_text(canonical_json(rows)),
        "fetched_unix_seconds": int(time.time()),
        "rows": rows,
    }
    atomic_write_json(cache_path, document)
    return rows


def format_options(question: str, options: list[str]) -> str:
    text = f"Question: {question}\nOptions: "
    for index, option in enumerate(options):
        text += f"{CHOICES[index]}. {option}\n"
    return text


def official_format_example(row: dict[str, Any], *, target: bool) -> str:
    cot_content = row["cot_content"]
    if target or not cot_content:
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    return format_options(row["question"], row["options"]) + f"Answer: {cot_content}\n\n"


def direct_format_example(row: dict[str, Any], *, target: bool) -> str:
    text = format_options(row["question"], row["options"])
    if target:
        return text + "Answer: The answer is ("
    return text + f"Answer: The answer is ({row['answer']}).\n\n"


def category_prefix(category: str, examples: list[dict[str, Any]], mode: str) -> str:
    if mode == "official-cot":
        prompt = (
            "The following are multiple choice questions (with answers) about "
            f"{category}. Think step by step and then output the answer in the format of "
            '"The answer is (X)" at the end.\n\n'
        )
        for example in examples:
            prompt += official_format_example(example, target=False)
        return prompt
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{category}. Choose the single best option. Respond only with "
        '"The answer is (X)" where X is its letter.\n\n'
    )
    for example in examples:
        prompt += direct_format_example(example, target=False)
    return prompt


def prepare_tasks(
    test_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    mode: str,
    categories: Optional[set[str]],
    question_ids: Optional[set[int]],
    limit: int,
    limit_per_category: int = 0,
) -> list[Task]:
    validation_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        validation_by_category[row["category"]].append(row)
    for row in test_rows:
        test_by_category[row["category"]].append(row)

    tasks: list[Task] = []
    sequence = 0
    for category in sorted(test_by_category):
        if categories is not None and category not in categories:
            continue
        examples = validation_by_category.get(category, [])
        if len(examples) != 5:
            raise EvalError(
                f"category {category!r} has {len(examples)} validation examples; expected 5"
            )
        shared_prefix = category_prefix(category, examples, mode)
        shared_prefix_sha256 = sha256_text(shared_prefix)
        selected_in_category = 0
        for row in test_by_category[category]:
            if question_ids is not None and row["question_id"] not in question_ids:
                continue
            if limit_per_category > 0 and selected_in_category >= limit_per_category:
                break
            target = (
                official_format_example(row, target=True)
                if mode == "official-cot"
                else direct_format_example(row, target=True)
            )
            prompt = shared_prefix + target
            tasks.append(
                Task(
                    sequence=sequence,
                    question_id=row["question_id"],
                    category=category,
                    source=row["src"],
                    answer=row["answer"],
                    prompt=prompt,
                    prompt_sha256=sha256_text(prompt),
                    shared_prefix_sha256=shared_prefix_sha256,
                )
            )
            sequence += 1
            selected_in_category += 1
            if limit > 0 and len(tasks) >= limit:
                return tasks
    return tasks


def extract_answer(text: str) -> Optional[str]:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None


def request_messages(task: Task, mode: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if mode == "direct":
        messages.append({"role": "system", "content": DIRECT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": task.prompt})
    return messages


def endpoint_headers(endpoint: Endpoint) -> dict[str, str]:
    if endpoint.api_key:
        return {"Authorization": f"Bearer {endpoint.api_key}"}
    return {}


def probe_endpoint(endpoint: Endpoint, timeout: float) -> dict[str, Any]:
    response = open_json_url(
        endpoint.models_url,
        headers=endpoint_headers(endpoint),
        timeout=timeout,
        direct=True,
    )
    model_ids = {
        str(item.get("id"))
        for item in response.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    if model_ids and endpoint.model not in model_ids:
        raise EvalError(
            f"{endpoint.side} endpoint exposes {sorted(model_ids)}, not {endpoint.model!r}"
        )
    return {"models": sorted(model_ids), "object": response.get("object")}


def retryable_error(error: BaseException) -> bool:
    if isinstance(error, HttpStatusError):
        return error.status == 429 or error.status >= 500
    return isinstance(error, EvalError)


def run_one(
    task: Task,
    endpoint: Endpoint,
    *,
    config_id: str,
    mode: str,
    max_completion_tokens: int,
    enable_thinking: bool,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    messages = request_messages(task, mode)
    payload = {
        "model": endpoint.model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    request_started = time.time()
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        attempt_started = time.perf_counter()
        try:
            response = open_json_url(
                endpoint.completion_url,
                method="POST",
                payload=payload,
                headers=endpoint_headers(endpoint),
                timeout=timeout,
                direct=True,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise EvalError("chat completion response contains no choices")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise EvalError("chat completion choice contains no message")
            message = choice["message"]
            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise EvalError("chat completion content is not a string")
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if not isinstance(reasoning, str):
                reasoning = str(reasoning)
            extraction_text = reasoning + content if enable_thinking else content
            prediction = extract_answer(extraction_text)
            elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
            return {
                "schema_version": 1,
                "record_type": "mmlu_pro_openai_result",
                "config_id": config_id,
                "endpoint": endpoint.public_identity,
                "sequence": task.sequence,
                "question_id": task.question_id,
                "category": task.category,
                "source": task.source,
                "answer": task.answer,
                "prediction": prediction,
                "correct": prediction == task.answer,
                "parsed": prediction is not None,
                "ok": True,
                "mode": mode,
                "prompt_sha256": task.prompt_sha256,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "shared_prefix_sha256": task.shared_prefix_sha256,
                "response_sha256": sha256_text(content),
                "content": content,
                "reasoning_content": reasoning,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "qrt_metrics": response.get("qrt_metrics"),
                "response_id": response.get("id"),
                "system_fingerprint": response.get("system_fingerprint"),
                "elapsed_ms": elapsed_ms,
                "attempts": attempt + 1,
                "started_unix_seconds": request_started,
                "completed_unix_seconds": time.time(),
            }
        except BaseException as error:
            last_error = error
            if attempt >= retries or not retryable_error(error):
                break
            time.sleep(retry_delay * (2**attempt))

    assert last_error is not None
    return {
        "schema_version": 1,
        "record_type": "mmlu_pro_openai_result",
        "config_id": config_id,
        "endpoint": endpoint.public_identity,
        "sequence": task.sequence,
        "question_id": task.question_id,
        "category": task.category,
        "source": task.source,
        "answer": task.answer,
        "prediction": None,
        "correct": False,
        "parsed": False,
        "ok": False,
        "mode": mode,
        "prompt_sha256": task.prompt_sha256,
        "messages_sha256": sha256_text(canonical_json(messages)),
        "shared_prefix_sha256": task.shared_prefix_sha256,
        "error_type": type(last_error).__name__,
        "error": str(last_error),
        "attempts": retries + 1,
        "started_unix_seconds": request_started,
        "completed_unix_seconds": time.time(),
    }


def iter_result_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvalError(f"invalid JSONL at {path}:{line_number}") from error
            if isinstance(record, dict):
                yield record


def successful_question_ids(path: Path, config_id: str, endpoint_id: str) -> set[int]:
    completed: set[int] = set()
    for record in iter_result_records(path):
        endpoint = record.get("endpoint") or {}
        if (
            record.get("config_id") == config_id
            and endpoint.get("endpoint_id") == endpoint_id
            and record.get("ok") is True
        ):
            completed.add(int(record["question_id"]))
    return completed


def run_endpoint(
    tasks: list[Task],
    endpoint: Endpoint,
    writer: ResultWriter,
    results_path: Path,
    *,
    config_id: str,
    mode: str,
    max_completion_tokens: int,
    enable_thinking: bool,
    timeout: float,
    retries: int,
    retry_delay: float,
    workers: int,
    progress_every: int,
    request_delay: float,
) -> None:
    completed_ids = successful_question_ids(results_path, config_id, endpoint.endpoint_id)
    pending = [task for task in tasks if task.question_id not in completed_ids]
    print(
        f"{endpoint.side}: endpoint_id={endpoint.endpoint_id} total={len(tasks)} "
        f"resumed={len(tasks) - len(pending)} pending={len(pending)} workers={workers}",
        flush=True,
    )
    if not pending:
        return

    progress_lock = threading.Lock()
    finished = 0
    correct = 0
    parsed = 0
    failed = 0
    started = time.perf_counter()

    def execute(task: Task) -> dict[str, Any]:
        if request_delay > 0:
            time.sleep(request_delay)
        return run_one(
            task,
            endpoint,
            config_id=config_id,
            mode=mode,
            max_completion_tokens=max_completion_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )

    def retain(record: dict[str, Any]) -> None:
        nonlocal finished, correct, parsed, failed
        writer.append(record)
        with progress_lock:
            finished += 1
            correct += int(record.get("correct") is True)
            parsed += int(record.get("parsed") is True)
            failed += int(record.get("ok") is not True)
            if (
                finished == 1
                or finished == len(pending)
                or finished % max(1, progress_every) == 0
            ):
                elapsed = time.perf_counter() - started
                rate = finished / elapsed if elapsed > 0 else 0.0
                print(
                    f"{endpoint.side}: {finished}/{len(pending)} "
                    f"correct={correct} parsed={parsed} failed={failed} "
                    f"rate={rate:.3f} question/s last_qid={record['question_id']}",
                    flush=True,
                )

    if workers == 1:
        for task in pending:
            retain(execute(task))
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            retain(future.result())


def summarize(
    tasks: list[Task],
    results_paths: list[Path],
    endpoints: list[Endpoint],
    config: dict[str, Any],
    runtime_evidence: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    expected = {task.question_id: task for task in tasks}
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    failures: dict[tuple[str, int], dict[str, Any]] = {}
    for results_path in results_paths:
        for record in iter_result_records(results_path):
            endpoint = record.get("endpoint") or {}
            endpoint_id = endpoint.get("endpoint_id")
            question_id = record.get("question_id")
            if (
                record.get("config_id") != config["config_id"]
                or endpoint_id is None
                or question_id is None
                or int(question_id) not in expected
            ):
                continue
            key = (str(endpoint_id), int(question_id))
            if record.get("ok") is True:
                latest[key] = record
            else:
                failures[key] = record

    endpoint_summaries: dict[str, Any] = {}
    endpoint_records: dict[str, dict[int, dict[str, Any]]] = {}
    for endpoint in endpoints:
        records = {
            question_id: latest[(endpoint.endpoint_id, question_id)]
            for question_id in expected
            if (endpoint.endpoint_id, question_id) in latest
        }
        endpoint_records[endpoint.side] = records
        correct = sum(record.get("correct") is True for record in records.values())
        parsed = sum(record.get("parsed") is True for record in records.values())
        by_category: dict[str, Any] = {}
        for category in sorted({task.category for task in tasks}):
            category_ids = [
                task.question_id for task in tasks if task.category == category
            ]
            category_records = [records[qid] for qid in category_ids if qid in records]
            category_correct = sum(
                record.get("correct") is True for record in category_records
            )
            by_category[category] = {
                "expected": len(category_ids),
                "completed": len(category_records),
                "correct": category_correct,
                "accuracy": (
                    category_correct / len(category_ids) if category_ids else 0.0
                ),
            }
        endpoint_summaries[endpoint.side] = {
            **endpoint.public_identity,
            "expected": len(tasks),
            "completed": len(records),
            "missing": len(tasks) - len(records),
            "failed_without_success": sum(
                (endpoint.endpoint_id, question_id) in failures
                and question_id not in records
                for question_id in expected
            ),
            "parsed": parsed,
            "unparsed": len(records) - parsed,
            "correct": correct,
            "accuracy": correct / len(tasks) if tasks else 0.0,
            "by_category": by_category,
        }

    parity: Optional[dict[str, Any]] = None
    if "candidate" in endpoint_records and "reference" in endpoint_records:
        candidate = endpoint_records["candidate"]
        reference = endpoint_records["reference"]
        common_ids = sorted(set(candidate).intersection(reference))
        prediction_equal = sum(
            candidate[qid].get("prediction") == reference[qid].get("prediction")
            for qid in common_ids
        )
        correctness_equal = sum(
            candidate[qid].get("correct") == reference[qid].get("correct")
            for qid in common_ids
        )
        candidate_correct = endpoint_summaries["candidate"]["correct"]
        reference_correct = endpoint_summaries["reference"]["correct"]
        both_full = (
            endpoint_summaries["candidate"]["completed"] == len(tasks)
            and endpoint_summaries["reference"]["completed"] == len(tasks)
        )
        parity = {
            "common_completed": len(common_ids),
            "prediction_equal": prediction_equal,
            "prediction_agreement": (
                prediction_equal / len(common_ids) if common_ids else 0.0
            ),
            "correctness_equal": correctness_equal,
            "correctness_agreement": (
                correctness_equal / len(common_ids) if common_ids else 0.0
            ),
            "candidate_only_correct": sum(
                candidate[qid].get("correct") is True
                and reference[qid].get("correct") is not True
                for qid in common_ids
            ),
            "reference_only_correct": sum(
                reference[qid].get("correct") is True
                and candidate[qid].get("correct") is not True
                for qid in common_ids
            ),
            "score_numerator_delta": candidate_correct - reference_correct,
            "score_percentage_point_delta": (
                (candidate_correct - reference_correct) * 100.0 / len(tasks)
                if tasks
                else 0.0
            ),
            "both_full": both_full,
            "score_equal": candidate_correct == reference_correct,
            "passed": both_full and candidate_correct == reference_correct,
        }

    return {
        "schema_version": 1,
        "record_type": "mmlu_pro_openai_summary",
        "generated_unix_seconds": int(time.time()),
        "config": config,
        "results_paths": [str(path.resolve()) for path in results_paths],
        "expected_questions": len(tasks),
        "endpoints": endpoint_summaries,
        "runtime_evidence": runtime_evidence or {},
        "parity": parity,
    }


def resolve_api_key(environment_name: Optional[str]) -> Optional[str]:
    if not environment_name:
        return None
    value = os.environ.get(environment_name)
    if not value:
        raise EvalError(f"API key environment variable {environment_name!r} is empty")
    return value


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full, resumable MMLU-Pro score parity over OpenAI chat APIs."
    )
    parser.add_argument(
        "--side",
        choices=("candidate", "reference", "both", "summarize"),
        default="candidate",
    )
    parser.add_argument(
        "--candidate-url", "--base-url", dest="candidate_url",
        default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument(
        "--candidate-model", "--model", dest="candidate_model",
        default="qwen3.6-35b-a3b"
    )
    parser.add_argument("--candidate-api-key-env")
    parser.add_argument(
        "--candidate-provenance-file", type=Path, action="append"
    )
    parser.add_argument(
        "--reference-url", default="https://reference.example.invalid/v1"
    )
    parser.add_argument("--reference-model", default="qwen3.6-35b-a3b")
    parser.add_argument("--reference-api-key-env")
    parser.add_argument(
        "--reference-provenance-file", type=Path, action="append"
    )
    parser.add_argument(
        "--mode", choices=("direct", "official-cot"), default="direct"
    )
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/eval/mmlu-pro-openai")
    )
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--results-file", "--output", dest="results_file", type=Path)
    parser.add_argument("--additional-results-file", type=Path, action="append")
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--category", action="append")
    parser.add_argument("--question-id", type=int, action="append")
    parser.add_argument("--question-id-file", type=Path, action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--fsync-every", type=int, default=1)
    parser.add_argument("--dataset-page-delay-seconds", type=float, default=2.0)
    parser.add_argument("--skip-endpoint-probe", action="store_true")
    parser.add_argument("--manifest-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.limit < 0:
        raise EvalError("--limit cannot be negative")
    if args.limit_per_category < 0:
        raise EvalError("--limit-per-category cannot be negative")
    if args.workers < 1:
        raise EvalError("--workers must be at least one")
    if args.retries < 0:
        raise EvalError("--retries cannot be negative")
    if args.dataset_page_delay_seconds < 0:
        raise EvalError("--dataset-page-delay-seconds cannot be negative")
    max_completion_tokens = args.max_completion_tokens
    if max_completion_tokens is None:
        max_completion_tokens = 8 if args.mode == "direct" else 512
    if max_completion_tokens < 1:
        raise EvalError("--max-completion-tokens must be at least one")

    output_dir = args.output_dir.resolve()
    dataset_cache = (args.dataset_cache or output_dir / "dataset").resolve()
    validation_rows = fetch_dataset_split(
        dataset_cache,
        "validation",
        args.dataset_revision,
        page_delay_seconds=args.dataset_page_delay_seconds,
    )
    test_rows = fetch_dataset_split(
        dataset_cache,
        "test",
        args.dataset_revision,
        page_delay_seconds=args.dataset_page_delay_seconds,
    )
    categories = set(args.category) if args.category else None
    file_question_ids, question_id_file_evidence = load_question_id_files(
        args.question_id_file or []
    )
    requested_question_ids = list(args.question_id or []) + file_question_ids
    if len(requested_question_ids) != len(set(requested_question_ids)):
        raise EvalError("duplicate question ID across command-line and file selection")
    question_ids = set(requested_question_ids) if requested_question_ids else None
    tasks = prepare_tasks(
        test_rows,
        validation_rows,
        mode=args.mode,
        categories=categories,
        question_ids=question_ids,
        limit=args.limit,
        limit_per_category=args.limit_per_category,
    )
    if question_ids is not None:
        selected_question_ids = {task.question_id for task in tasks}
        missing_question_ids = sorted(question_ids - selected_question_ids)
        if missing_question_ids:
            raise EvalError(
                "requested question IDs are absent from the selected dataset/categories: "
                + ", ".join(str(value) for value in missing_question_ids)
            )
    if not tasks:
        raise EvalError("no benchmark questions were selected")

    prompt_version = (
        PROMPT_VERSION if args.mode == "official-cot" else DIRECT_PROMPT_VERSION
    )
    config_material = {
        "dataset": DATASET_ID,
        "dataset_revision": args.dataset_revision,
        "official_harness_revision": OFFICIAL_HARNESS_REVISION,
        "mode": args.mode,
        "prompt_version": prompt_version,
        "system_prompt": DIRECT_SYSTEM_PROMPT if args.mode == "direct" else None,
        "shots": 5,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": args.enable_thinking,
        "max_completion_tokens": max_completion_tokens,
        "client_concurrency": args.workers,
    }
    config_id = sha256_text(canonical_json(config_material))[:16]
    config = {**config_material, "config_id": config_id}

    results_path = (
        args.results_file or output_dir / f"results-{config_id}.jsonl"
    ).resolve()
    summary_results_paths = [results_path]
    for additional_path in args.additional_results_file or []:
        resolved = additional_path.resolve()
        if resolved not in summary_results_paths:
            summary_results_paths.append(resolved)
    if args.summary_file is not None:
        summary_path = args.summary_file.resolve()
    elif args.results_file is not None:
        summary_path = results_path.with_name(
            f"{results_path.stem}.summary.json"
        )
    else:
        summary_path = (output_dir / f"summary-{config_id}.json").resolve()
    if args.manifest_file is not None:
        manifest_path = args.manifest_file.resolve()
    elif args.results_file is not None:
        manifest_path = results_path.with_name(
            f"{results_path.stem}.manifest.json"
        )
    else:
        manifest_path = (output_dir / f"manifest-{config_id}.json").resolve()

    candidate = Endpoint(
        side="candidate",
        base_url=args.candidate_url,
        model=args.candidate_model,
        api_key=resolve_api_key(args.candidate_api_key_env),
    )
    reference = Endpoint(
        side="reference",
        base_url=args.reference_url,
        model=args.reference_model,
        api_key=resolve_api_key(args.reference_api_key_env),
    )
    selected_endpoints = (
        [candidate]
        if args.side == "candidate"
        else [reference]
        if args.side == "reference"
        else [reference, candidate]
        if args.side == "both"
        else [candidate, reference]
    )
    runtime_evidence = {
        "candidate": load_runtime_evidence(args.candidate_provenance_file or []),
        "reference": load_runtime_evidence(args.reference_provenance_file or []),
    }

    manifest = {
        "schema_version": 1,
        "record_type": "mmlu_pro_openai_manifest",
        "config": config,
        "selected_questions": len(tasks),
        "selected_categories": sorted({task.category for task in tasks}),
        "selection": {
            "categories": sorted(categories) if categories is not None else None,
            "question_ids": sorted(question_ids) if question_ids is not None else None,
            "question_id_files": question_id_file_evidence or None,
            "limit": args.limit,
            "limit_per_category": args.limit_per_category,
        },
        "execution": {
            "workers": args.workers,
            "timeout_seconds": args.timeout_seconds,
            "retries": args.retries,
            "retry_delay_seconds": args.retry_delay_seconds,
            "request_delay_seconds": args.request_delay_seconds,
            "fsync_every": args.fsync_every,
        },
        "test_rows_sha256": sha256_text(canonical_json(test_rows)),
        "validation_rows_sha256": sha256_text(canonical_json(validation_rows)),
        "endpoints": [endpoint.public_identity for endpoint in selected_endpoints],
        "runtime_evidence": runtime_evidence,
        "results_path": str(results_path),
        "summary_path": str(summary_path),
    }
    atomic_write_json(manifest_path, manifest)

    if args.side != "summarize":
        if not args.skip_endpoint_probe:
            for endpoint in selected_endpoints:
                probe = probe_endpoint(endpoint, min(args.timeout_seconds, 60.0))
                print(
                    f"{endpoint.side}: endpoint={endpoint.base_url} "
                    f"model={endpoint.model} probe={canonical_json(probe)}",
                    flush=True,
                )
        with ResultWriter(results_path, args.fsync_every) as writer:
            for endpoint in selected_endpoints:
                run_endpoint(
                    tasks,
                    endpoint,
                    writer,
                    results_path,
                    config_id=config_id,
                    mode=args.mode,
                    max_completion_tokens=max_completion_tokens,
                    enable_thinking=args.enable_thinking,
                    timeout=args.timeout_seconds,
                    retries=args.retries,
                    retry_delay=args.retry_delay_seconds,
                    workers=args.workers,
                    progress_every=args.progress_every,
                    request_delay=args.request_delay_seconds,
                )

    summary = summarize(
        tasks,
        summary_results_paths,
        [candidate, reference],
        config,
        runtime_evidence,
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    parity = summary.get("parity")
    if args.side == "summarize" and parity is not None and parity.get("passed"):
        return 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("evaluation interrupted; completed JSONL records are resumable", file=sys.stderr)
        raise SystemExit(130)
    except EvalError as error:
        print(f"eval error: {error}", file=sys.stderr)
        raise SystemExit(2)
