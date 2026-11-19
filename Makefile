#Compiler 
UV := uv

#Sources
SRCS := -m src

# MyPy Flags
MYPY_FLGS := --warn-return-any \
			--warn-unused-ignores \
			--ignore-missing-imports \
			--disallow-untyped-defs \
			--check-untyped-defs

# Dependencies and cache location
CMM_DIR := $(HOME)/goinfre/call-me-maybe

UV_CACHE_DIR := $(CMM_DIR)/uv/uv-cache
UV_PROJECT_ENVIRONMENT := $(CMM_DIR)/uv/venvs/call-me-maybe

HF_HUB_CACHE := $(CMM_DIR)/huggingface/hub
TRANSFORMERS_CACHE := $(CMM_DIR)/huggingface/transformers
OUTPUT_DIR := ./data/output

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT
export HF_HUB_CACHE
export TRANSFORMERS_CACHE

# Terminal Colors
RESET       = \033[0m
PINK        = \033[1;35m
BLUE        = \033[1;36m
GREEN       = \033[1;32m

.PHONY: install run debug clean lint

all: install run

install:
	mkdir -p $(UV_CACHE_DIR)
	mkdir -p $(UV_PROJECT_ENVIRONMENT)
	mkdir -p $(HF_HUB_CACHE)
	mkdir -p $(TRANSFORMERS_CACHE)
	mkdir -p $(OUTPUT_DIR)
	$(UV) sync

run:
	mkdir -p $(OUTPUT_DIR)
	$(UV) run python $(SRCS)
	
debug:
	$(UV) run python -m pdb $(SRCS)

clean: 
	rm -rf .pytest_cache .mypy_cache __pycache__ src/__pycache__ $(CMM_DIR) $(OUTPUT_DIR)
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Objects cleaned.$(RESET)"

lint:
	$(UV) run flake8 src/
	$(UV) run mypy src/ $(MYPY_FLGS)

lint-strict:
	$(UV) run flake8 src/
	$(UV) run mypy --strict src/