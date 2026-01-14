install:
	pip install -r requirements.txt
train:
	python train.py
test:
	pytest
build:
	docker build -t ml-ci-cd-demo .
run:
	docker run -p 8000:8000 ml-ci-cd-demo