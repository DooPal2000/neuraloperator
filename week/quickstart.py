from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import Trainer
import torch
from neuralop import LpLoss, H1Loss


def main():
    # 1. 데이터 로드
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000,
        batch_size=32,
        test_resolutions=[16, 32],
        n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )

    # 2. 모델 생성
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1)

    # 3. 옵티마이저 & 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    # 4. 학습
    trainer = Trainer(
        model=model,
        n_epochs=20,
        device="cpu",
        data_processor=data_processor,
        eval_interval=3,
        verbose=True,  # ← 콘솔 출력을 켜는 옵션
        log_output=True,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses={"h1": h1loss, "l2": l2loss},
    )


if __name__ == "__main__":
    main()
