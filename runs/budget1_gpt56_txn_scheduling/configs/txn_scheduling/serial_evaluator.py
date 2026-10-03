import fcntl
import runpy
import sys

sys.path.insert(0, '/home/chenhui/efficient-evolve/benchmarks/ADRS/txn_scheduling/evaluator')
_original = runpy.run_path('/home/chenhui/efficient-evolve/benchmarks/ADRS/txn_scheduling/evaluator/evaluator.py')

def evaluate(program_path):
    with open('/tmp/frugalevo-adrs-1007.lock', "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _original["evaluate"](program_path)

def evaluate_stage1(program_path):
    return evaluate(program_path)

def evaluate_stage2(program_path):
    return evaluate(program_path)

def evaluate_stage3(program_path):
    return evaluate(program_path)
