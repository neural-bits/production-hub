
.SILENT:

RED:=\033[0;31m
GREEN:=\033[0;32m
BLUE:=\033[0;34m
NC:=\033[0m

# Set the default shell to bash
SHELL := /bin/bash

help:
	@echo -e "$(BLUE)Usage: make [target]$(NC)"
	@echo -e "$(BLUE)Targets:$(NC)"
	@echo -e "$(BLUE)  start_tis$(NC) - Start TIS"
	@echo -e "$(BLUE)  stop_tis$(NC) - Stop TIS"
	@echo -e "$(BLUE)  install$(NC) - Install dependencies"
	@echo -e "$(BLUE)  install_nvidia_container_runtime$(NC) - Install NVIDIA Container Runtime"
	

install_nvidia_container_runtime:
	@echo -e "$(GREEN)Installing NVIDIA Container Runtime...$(NC)"
	@curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
	&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
	sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
	sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	@sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
	@sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
	sudo systemctl restart docker
	@echo -e "$(GREEN)NVIDIA Container Runtime installed$(NC)"


install:
	@echo -e "$(GREEN)Installing dependencies...$(NC)"
	poetry init \
	&& export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	&& poetry install --no-root
	@echo -e "$(GREEN)Dependencies installed$(NC)"
	@echo -e "$(GREEN)Installing Jupyter kernel...$(NC)"
	poetry run ipython kernel install --user --name=py311_inference_engines