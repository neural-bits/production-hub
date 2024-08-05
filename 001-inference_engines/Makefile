
install:
	poetry init \
	&& export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
	&& poetry install --no-root \
	&& poetry run ipython kernel install --user --name=py311_inference_engines