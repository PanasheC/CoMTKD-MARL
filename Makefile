.PHONY: install test smoke clean

install:
	python -m pip install -r requirements.txt

test:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest

smoke:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_student_rl.py \
	  --teacher-name-list resnet20 resnet32 \
	  --allow-random-teachers --dummy-data \
	  --dummy-train-size 8 --dummy-test-size 4 --batch-size 4 \
	  --epochs 1 --workers 0 --arch resnet20 --common-dim 16 \
	  --hidden-dim 16 --attention-heads 4 --sync-rounds 1 \
	  --rollout-size 4 --ppo-minibatch-size 4 --ppo-epochs 1 \
	  --reward-mode negative_loss --device cpu --checkpoint-dir ./_smoke

clean:
	rm -rf .pytest_cache _smoke **/__pycache__
