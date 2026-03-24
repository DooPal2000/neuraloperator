"""
나비에-스토크 방정식 데이터셋 로드 및 용량 확인
"""

from pathlib import Path
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt


def main():
    # 데이터 디렉토리 설정
    data_dir = Path("~/data/navier_stokes/").expanduser()

    print(f"데이터 디렉토리: {data_dir}")
    print(f"디렉토리 존재 여부: {data_dir.exists()}")

    # 1. 데이터 로드 (다운로드 필요 시 자동 다운로드)
    print("\n데이터 로드 중... (최초 실행 시 다운로드 필요)")
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=128,
        n_train=1000,  # 테스트용으로 1000개만
        batch_size=8,
        test_resolutions=[128],
        n_tests=[100],
        test_batch_sizes=[8],
        encode_input=True,
        encode_output=True,
    )

    # 2. 데이터셋 정보 출력
    print("\n" + "=" * 50)
    print("데이터셋 정보")
    print("=" * 50)
    print(f"훈련 데이터셋 크기: {len(train_loader.dataset)}")
    print(f"테스트 데이터셋 크기: {len(test_loaders[128].dataset)}")
    print(f"배치 사이즈: {train_loader.batch_size}")
    print(f"데이터 프로세서: {type(data_processor).__name__}")

    # 3. 데이터 형태 확인
    batch = next(iter(train_loader))
    print(f"\n배치 형태:")
    print(f"  입력 (x): {batch['x'].shape}")
    print(f"  출력 (y): {batch['y'].shape}")

    # 4. 다운로드된 파일 용량 확인
    print("\n" + "=" * 50)
    print("다운로드된 파일 목록 및 용량")
    print("=" * 50)

    total_size = 0
    if data_dir.exists():
        for file in sorted(data_dir.glob("*.pt")):
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += file.stat().st_size
            print(f"{file.name:40} {size_mb:8.2f} MB")

        # 압축 파일도 확인
        for file in sorted(data_dir.glob("*.tgz")):
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += file.stat().st_size
            print(f"{file.name:40} {size_mb:8.2f} MB")

        print("=" * 50)
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"총 용량: {total_size_gb:.2f} GB")
        print("=" * 50)

    return train_loader, test_loaders, data_processor


if __name__ == "__main__":
    train_loader, test_loaders, data_processor = main()
