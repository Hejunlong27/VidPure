"""批量队列：上限校验、超时长跳过、并发、失败重试（不依赖 GPU / FFmpeg）。

关键做法：monkeypatch `batch.ffprobe` 与 `BatchRunner._run_job`，
把「探测时长」与「实际处理」替换成可控的假实现，从而只测队列逻辑本身。
"""

import time
from pathlib import Path

import pytest

from vidpure.core import batch
from vidpure.core.batch import (BatchError, BatchLimits, BatchRunner, JobOptions,
                                _merge_opts, _weights)

FAKE_META = {"duration": 10.0, "w": 640, "h": 360, "fps": 24.0,
             "has_audio": True, "audio_codec": "aac"}


@pytest.fixture
def patch_ffprobe(monkeypatch):
    monkeypatch.setattr(batch, "ffprobe", lambda p: dict(FAKE_META))


def _make(tmp_path: Path, n: int = 1, ext: str = ".mp4") -> list[Path]:
    out = []
    for i in range(n):
        f = tmp_path / f"v{i}{ext}"
        f.write_bytes(b"x")
        out.append(f)
    return out


def _wait(runner: BatchRunner, ids, timeout: float = 20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(runner.job(i).status in ("done", "error", "rejected") for i in ids):
            return True
        time.sleep(0.03)
    return False


# ------------------------------------------------------------------ 提交校验
def test_reject_more_than_max_items(tmp_path, patch_ffprobe):
    r = BatchRunner(out_dir=tmp_path / "o", limits=BatchLimits(max_items=3))
    files = _make(tmp_path, 4)
    with pytest.raises(BatchError) as e:
        r.submit([{"src": str(f)} for f in files])
    assert "最多" in str(e.value)


def test_sixty_items_accepted(tmp_path, patch_ffprobe, monkeypatch):
    """需求硬指标：一次性提交 60 个视频必须被接受。"""
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o", concurrency=4, retry_backoff=0.0)
    files = _make(tmp_path, 60)
    jobs = r.submit([{"src": str(f)} for f in files])
    assert len(jobs) == 60
    assert _wait(r, [j.jid for j in jobs], 30)
    assert all(j.status == "done" for j in jobs)


def test_over_duration_item_is_rejected_but_batch_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ffprobe",
                        lambda p: {**FAKE_META, "duration": 200.0})
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o", retry_backoff=0.0)
    files = _make(tmp_path, 2)          # 每个都是 200s ⇒ 全部超过 90s 上限
    jobs = r.submit([{"src": str(f)} for f in files])
    assert _wait(r, [j.jid for j in jobs], 10)
    assert all(j.status == "rejected" for j in jobs)
    assert "上限" in jobs[0].error


def test_duration_limit_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ffprobe", lambda p: {**FAKE_META, "duration": 500.0})
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o",
                    limits=BatchLimits(enforce_limits=False), retry_backoff=0.0)
    jobs = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])
    assert _wait(r, [jobs[0].jid], 10)
    assert jobs[0].status == "done"


def test_missing_file_is_rejected(tmp_path, patch_ffprobe):
    r = BatchRunner(out_dir=tmp_path / "o")
    jobs = r.submit([{"src": str(tmp_path / "nope.mp4")}])
    assert jobs[0].status == "rejected"
    assert "不存在" in jobs[0].error


def test_empty_submission_raises(tmp_path):
    r = BatchRunner(out_dir=tmp_path / "o")
    with pytest.raises(BatchError):
        r.submit([])


# ------------------------------------------------------------------ 重试
def test_retries_then_succeeds(tmp_path, patch_ffprobe, monkeypatch):
    r = BatchRunner(out_dir=tmp_path / "o", retries=2, retry_backoff=0.05)
    calls = {"n": 0}

    def flaky(self, job):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("模拟失败")

    monkeypatch.setattr(BatchRunner, "_run_job", flaky)
    j = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])[0]
    assert _wait(r, [j.jid], 20)
    assert calls["n"] == 3 and j.status == "done" and j.attempts == 3


def test_exhausts_retries_then_errors(tmp_path, patch_ffprobe, monkeypatch):
    r = BatchRunner(out_dir=tmp_path / "o", retries=1, retry_backoff=0.05)
    calls = {"n": 0}

    def always_fail(self, job):
        calls["n"] += 1
        raise RuntimeError("总是失败")

    monkeypatch.setattr(BatchRunner, "_run_job", always_fail)
    j = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])[0]
    assert _wait(r, [j.jid], 20)
    assert calls["n"] == 2 and j.status == "error" and j.error


def test_failure_of_one_job_does_not_block_others(tmp_path, patch_ffprobe, monkeypatch):
    """单条失败不得影响整批（原 skill「某集炸掉整批」的教训）。"""
    r = BatchRunner(out_dir=tmp_path / "o", concurrency=2, retries=0, retry_backoff=0.0)

    def selective(self, job):
        if Path(job.src).name == "v1.mp4":
            raise RuntimeError("坏文件")

    monkeypatch.setattr(BatchRunner, "_run_job", selective)
    jobs = r.submit([{"src": str(f)} for f in _make(tmp_path, 4)])
    assert _wait(r, [j.jid for j in jobs], 20)
    statuses = {Path(j.src).name: j.status for j in jobs}
    assert statuses["v1.mp4"] == "error"
    assert sum(1 for s in statuses.values() if s == "done") == 3


def test_manual_retry_resets_failed_job(tmp_path, patch_ffprobe, monkeypatch):
    r = BatchRunner(out_dir=tmp_path / "o", retries=0, retry_backoff=0.0)
    state = {"fail": True}

    def once(self, job):
        if state["fail"]:
            raise RuntimeError("先失败")

    monkeypatch.setattr(BatchRunner, "_run_job", once)
    j = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])[0]
    assert _wait(r, [j.jid], 10) and j.status == "error"
    state["fail"] = False
    r.retry(j.jid)
    assert _wait(r, [j.jid], 10)
    assert j.status == "done"


def test_retry_rejects_unfinished_job(tmp_path, patch_ffprobe, monkeypatch):
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o", retry_backoff=0.0)
    j = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])[0]
    assert _wait(r, [j.jid], 10)
    with pytest.raises(BatchError):
        r.retry(j.jid)          # 已完成 ⇒ 不允许重试


# ------------------------------------------------------------------ 快照 / 并发
def test_snapshot_counts(tmp_path, patch_ffprobe, monkeypatch):
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o", concurrency=3, retries=1, retry_backoff=0.0)
    jobs = r.submit([{"src": str(f)} for f in _make(tmp_path, 2)])
    _wait(r, [j.jid for j in jobs], 10)
    snap = r.snapshot()
    assert snap["total"] == 2 and snap["counts"].get("done") == 2
    assert snap["concurrency"] == 3 and snap["retries"] == 1
    assert snap["jobs"][0]["video_url"].startswith("/api/result/")


def test_set_concurrency(tmp_path):
    r = BatchRunner(out_dir=tmp_path / "o", concurrency=1)
    r.set_concurrency(4)
    assert r.concurrency == 4
    r.set_concurrency(0)             # 下限保护
    assert r.concurrency == 1


def test_persist_writes_json(tmp_path, patch_ffprobe, monkeypatch):
    monkeypatch.setattr(BatchRunner, "_run_job", lambda self, j: None)
    r = BatchRunner(out_dir=tmp_path / "o", retry_backoff=0.0)
    j = r.submit([{"src": str(f)} for f in _make(tmp_path, 1)])[0]
    _wait(r, [j.jid], 10)
    p = r.persist()
    assert p.exists() and '"jobs"' in p.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 纯函数
def test_weights_sum_to_one():
    w = _weights([True, True, True])
    assert abs(sum(w) - 1.0) < 1e-9
    assert sum(_weights([True, False, False])) == 1.0
    assert _weights([False, False, False]) == [0.0, 0.0, 0.0]


def test_merge_opts_only_known_keys_and_box_tuple():
    base = JobOptions()
    got = _merge_opts(base, {"clean": "crop", "target_res": "1080p",
                             "sub_box": [0.1, 0.2, 0.3, 0.4], "bogus": 1, "src": "x"})
    assert got.clean == "crop" and got.target_res == "1080p"
    assert got.sub_box == (0.1, 0.2, 0.3, 0.4)
    assert not hasattr(got, "bogus")


def test_friendly_error_translates_common_causes():
    assert "whisper" in batch._friendly(RuntimeError("未找到 whisper-cli: x"))
    assert "iopaint" in batch._friendly(RuntimeError("No module named 'iopaint'"))
    assert batch._friendly(RuntimeError("其它错误")).startswith("处理失败")
