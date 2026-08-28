# GDSD edge detector -- local quadratic surface fit + zero-crossing
PYTHON ?= python3
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17
# ใช้ pkg-config ถ้ามี opencv4; fallback เป็น -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgcodecs -lopencv_imgproc")

.PHONY: setup demo test clean build-cpp test-cpp

setup:            ## Install the package in editable mode with dev tools
	$(PYTHON) -m pip install -e ".[dev]"

demo:             ## Run the demo on the generated sample image
	$(PYTHON) demo.py

test:             ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

build-cpp:        ## Build the native C++ CLI (needs OpenCV dev headers)
	mkdir -p src/cpp/build
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) src/cpp/gdsd_cli.cpp \
		-o src/cpp/build/gdsd_cli $(OPENCV_LIBS)

test-cpp: build-cpp  ## Differential test: C++ output must match Python exactly
	$(PYTHON) src/cpp/diff_test.py

clean:            ## Remove generated files
	rm -rf output/ .pytest_cache/ .coverage htmlcov/
	rm -rf src/cpp/build/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
