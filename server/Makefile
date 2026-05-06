build:
	docker build --build-context mem0_local=../mem0 -t mem0-api-server .

run_local:
	docker run -p 8000:8000 \
		-v $(shell pwd):/app \
		-v $(shell pwd)/../mem0:/app/mem0 \
		mem0-api-server --env-file .env

.PHONY: build run_local