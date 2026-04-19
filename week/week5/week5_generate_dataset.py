# Week 5: generate_dataset.py
# 가변 점도 실험을 위한 커스텀 나비에-스토크스 데이터셋 생성기.

# 예상 출력 구조:
#   data/navier_stokes_week5/
#     raw/
#       sample_000000.pt
#       ...
#     metadata.json

# 설명:
# - `solve_navier_stokes_vorticity` 내부는 기존에 사용 중인 해석기(solver) 호출로 교체해야 한다.
# - raw 샘플들은 정규화(normalization) 없이 저장된다.
# - 학습용 입력 텐서는 여기서 만들지 않는다. 이는 preprocess_dataset.py에서 처리한다.
from __future__ import annotations


import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ns_solver import NavierStokes2DVorticitySolver, NSSolverConfig

@dataclass
class SampleMetadata:
    sample_id: int
    file: str
    nu: float
    seed: int
    resolution: int
    t_final: float
    dt: float
    status: str
    message: str = ""


def random_vorticity_field(
    resolution: int,
    rng: np.random.Generator,
    n_modes_min: int = 3,
    n_modes_max: int = 6,
    k_max: int = 6,
    noise_scale: float = 0.05,
    normalize: bool = True,
) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, resolution, endpoint=False, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi, resolution, endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    field = np.zeros((resolution, resolution), dtype=np.float32)
    n_terms = int(rng.integers(n_modes_min, n_modes_max + 1))

    for _ in range(n_terms):
        kx = int(rng.integers(1, k_max + 1))
        ky = int(rng.integers(1, k_max + 1))
        a = float(rng.normal())
        b = float(rng.normal())
        phase1 = float(rng.uniform(0, 2 * np.pi))
        phase2 = float(rng.uniform(0, 2 * np.pi))

        field += a * np.sin(kx * X + ky * Y + phase1)
        field += b * np.cos(kx * X - ky * Y + phase2)

    if noise_scale > 0:
        field += noise_scale * rng.normal(size=(resolution, resolution)).astype(np.float32)

    if normalize:
        field = field - field.mean()
        field = field / (field.std() + 1e-6)

    return field.astype(np.float32)


def solve_navier_stokes_vorticity(
    w0: np.ndarray,
    nu: float,
    t_final: float,
    dt: float,
    **solver_kwargs,
) -> np.ndarray:
    config = NSSolverConfig(
        resolution=w0.shape[0],
        viscosity=nu,
        dt=dt,
        t_final=t_final,
        dealias=True,
    )
    solver = NavierStokes2DVorticitySolver(config)
    return solver.solve(initial_vorticity=w0)


def generate_single_sample(
    sample_id: int,
    nu: float,
    resolution: int,
    t_final: float,
    dt: float,
    seed: int,
    raw_dir: Path,
    **solver_kwargs: Any,
) -> SampleMetadata:
    rng = np.random.default_rng(seed)
    file_name = f"sample_{sample_id:06d}.pt"
    file_path = raw_dir / file_name

    try:
        w0 = random_vorticity_field(resolution=resolution, rng=rng)
        wt = solve_navier_stokes_vorticity(
            w0=w0,
            nu=nu,
            t_final=t_final,
            dt=dt,
            **solver_kwargs,
        )

        sample = {
            "x0": torch.tensor(w0, dtype=torch.float32),
            "y": torch.tensor(wt, dtype=torch.float32),
            "nu": float(nu),
            "seed": int(seed),
            "resolution": int(resolution),
            "t_final": float(t_final),
            "dt": float(dt),
        }
        torch.save(sample, file_path)

        return SampleMetadata(
            sample_id=sample_id,
            file=file_name,
            nu=float(nu),
            seed=int(seed),
            resolution=int(resolution),
            t_final=float(t_final),
            dt=float(dt),
            status="ok",
        )

    except Exception as e:
        return SampleMetadata(
            sample_id=sample_id,
            file=file_name,
            nu=float(nu),
            seed=int(seed),
            resolution=int(resolution),
            t_final=float(t_final),
            dt=float(dt),
            status="failed",
            message=str(e),
        )


def save_metadata(metadata: list[SampleMetadata], output_path: Path) -> None:
    payload = [asdict(m) for m in metadata]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_dataset(
    output_root: str | Path = "data/navier_stokes_week5",
    resolution: int = 128,
    nu_values: list[float] | None = None,
    n_samples_per_nu: int = 300,
    t_final: float = 1.0,
    dt: float = 1e-3,
    seed_base: int = 20260417,
    sample_id_offset: int = 0,
    **solver_kwargs: Any,
) -> list[SampleMetadata]:
    output_root = Path(output_root)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if nu_values is None:
        nu_values = np.geomspace(1e-3, 1e-2, 6).tolist()

    metadata: list[SampleMetadata] = []
    sample_id = sample_id_offset

    print("=" * 72)
    print("Generating Navier-Stokes Week 5 dataset")
    print("=" * 72)
    print(f"output_root       : {output_root}")
    print(f"resolution        : {resolution}")
    print(f"nu_values         : {[float(v) for v in nu_values]}")
    print(f"n_samples_per_nu  : {n_samples_per_nu}")
    print(f"t_final           : {t_final}")
    print(f"dt                : {dt}")
    print(f"total samples     : {len(nu_values) * n_samples_per_nu}")
    print("=" * 72)

    for nu_idx, nu in enumerate(nu_values):
        print(f"[nu {nu_idx + 1}/{len(nu_values)}] nu={nu:.6f}")
        for local_idx in range(n_samples_per_nu):
            seed = seed_base + sample_id
            meta = generate_single_sample(
                sample_id=sample_id,
                nu=float(nu),
                resolution=resolution,
                t_final=t_final,
                dt=dt,
                seed=seed,
                raw_dir=raw_dir,
                **solver_kwargs,
            )
            metadata.append(meta)
            sample_id += 1

            if (local_idx + 1) % 25 == 0 or (local_idx + 1) == n_samples_per_nu:
                ok_count = sum(m.status == "ok" for m in metadata)
                fail_count = sum(m.status == "failed" for m in metadata)
                print(
                    f"  generated {local_idx + 1:4d}/{n_samples_per_nu} | "
                    f"ok={ok_count:4d}, failed={fail_count:4d}"
                )

    save_metadata(metadata, output_root / "metadata.json")

    ok_count = sum(m.status == "ok" for m in metadata)
    fail_count = sum(m.status == "failed" for m in metadata)
    print("=" * 72)
    print(f"Done. ok={ok_count}, failed={fail_count}")
    print(f"metadata saved to: {output_root / 'metadata.json'}")
    print("=" * 72)

    return metadata


if __name__ == "__main__":
    generate_dataset(
        output_root="data/navier_stokes_week5",
        resolution=128,          # 처음엔 64로 테스트 (128은 느림)
        nu_values=np.geomspace(1e-3, 1e-2, 6).tolist(),
        n_samples_per_nu=500,     # 먼저 5개로 동작 확인
        t_final=1.0,
        dt=1e-3,
        sample_id_offset=750,
        seed_base=20260411 + 750,
        # solver= 줄 삭제
    )