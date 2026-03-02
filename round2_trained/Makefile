PYTHON ?= python3

.PHONY: train train-fast eval demo play test clean

# Full training: 4.9M-param model, ~2 hours on GPU
train:
	$(PYTHON) train.py

# Fast training: smaller model, ~30 minutes
train-fast:
	$(PYTHON) train.py --d-model 128 --d-ff 512 --total-steps 40000

# Evaluate trained model
eval:
	$(PYTHON) eval.py

# Run impressive demos (Fibonacci, multiplication, division, sqrt)
demo:
	$(PYTHON) demo.py

# Interactive REPL — step through programs with the neural computer
play:
	$(PYTHON) play.py

# Smoke test: verify package imports and interpreter correctness
test:
	$(PYTHON) -c "\
	from subleq import *; \
	import random; random.seed(0); \
	m,pc,r = make_multiply(7,9); \
	mem,_,_ = run(m,pc); \
	assert mem[r] == 63, f'multiply failed: {mem[r]}'; \
	m,pc,ra,rb = make_fibonacci(3); \
	mem,_,_ = run(m,pc); \
	assert mem[ra] == 8 and mem[rb] == 13, f'fib failed: {mem[ra]},{mem[rb]}'; \
	m,pc,r = make_div(100,7); \
	mem,_,_ = run(m,pc); \
	assert mem[r] == 14, f'div failed: {mem[r]}'; \
	m,pc,r = make_isqrt(81); \
	mem,_,_ = run(m,pc); \
	assert mem[r] == 9, f'isqrt failed: {mem[r]}'; \
	model = MiniSUBLEQTransformer(d_model=64, n_heads=4, n_layers=4, d_ff=256); \
	import torch; x = torch.randint(0,256,(2,33)); y = model(x); \
	assert y.shape == (2,33,256), f'model shape: {y.shape}'; \
	print('All tests passed.'); \
	"

# Remove generated files
clean:
	rm -rf checkpoints/ __pycache__ subleq/__pycache__
