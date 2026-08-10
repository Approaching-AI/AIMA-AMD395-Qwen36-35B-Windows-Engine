CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O2 -Inative/src
LDLIBS ?= -lm
PYTHON ?= python3
BUILD_DIR ?= build
QRT_C_SRCS := native/src/qrt.c native/src/qwen36_baseline.c
QRT_C_HDRS := native/src/qrt.h native/src/qwen36_baseline.h

.PHONY: all check c-smoke rust-test python-test public-hygiene clean

all: check

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/qrt-c-smoke: $(QRT_C_SRCS) $(QRT_C_HDRS) native/src/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(QRT_C_SRCS) native/src/main.c -o $@ $(LDLIBS)

c-smoke: $(BUILD_DIR)/qrt-c-smoke
	./$(BUILD_DIR)/qrt-c-smoke

rust-test:
	cargo test --workspace
	cargo clippy --workspace --all-targets -- -D warnings

python-test:
	$(PYTHON) -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
	$(PYTHON) scripts/test_q16_transaction_contract.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

public-hygiene:
	$(PYTHON) tools/public_hygiene.py .

check: c-smoke rust-test python-test public-hygiene

clean:
	@build_dir='$(abspath $(BUILD_DIR))'; target_dir='$(CURDIR)/target'; \
	case "$$build_dir" in '$(CURDIR)'/*) ;; \
	  *) echo "refusing BUILD_DIR outside the repository: $$build_dir" >&2; exit 2 ;; \
	esac; \
	rm -rf -- "$$build_dir" "$$target_dir"
