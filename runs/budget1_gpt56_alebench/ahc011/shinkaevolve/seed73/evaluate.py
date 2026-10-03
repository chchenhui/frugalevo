"""Local ALE-Bench execution and official rank/performance conversion (no daemon)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ('ahc008', 'ahc011', 'ahc015', 'ahc016', 'ahc024', 'ahc025', 'ahc026', 'ahc027', 'ahc039', 'ahc046')
DEFAULT_DATA = ROOT / 'openevolve_output/ale_local_suite_data'
ALE_COMMIT = '3da9b12fb5d112dabb3af693d1a42031c95142bc'
PROJECT_CXX = ROOT / '.toolchains/gcc12/bin/x86_64-conda-linux-gnu-c++'


def cxx_compiler():
    """Use an explicit compiler override or the project-local C++20 toolchain."""
    override = os.environ.get('ALE_CXX')
    if override:
        return override
    return str(PROJECT_CXX) if PROJECT_CXX.is_file() else 'g++'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(problem, data_root, archive_dir=None):
    """Download/build official tools and generate public/private inputs once."""
    directory = Path(data_root).resolve() / problem
    if (directory / 'ready.json').exists():
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    archive = directory.parent / f'{problem}.zip'
    if not archive.exists():
        if archive_dir:
            shutil.copyfile(Path(archive_dir) / archive.name, archive)
        else:
            part = archive.with_suffix('.zip.part')
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/SakanaAI/ALE-Bench/resolve/main/{problem}.zip', part)
            part.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (directory.parent / member.filename).resolve()
            if not target.is_relative_to(directory):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        z.extractall(directory.parent)
    subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path', str(directory / 'tools/Cargo.toml')], check=True)
    metadata = json.loads((directory / 'data.json').read_text())
    for split in ('public', 'private'):
        seeds = metadata['seeds'][split]
        seedfile = directory / f'{split}_seeds.txt'
        seedfile.write_text(''.join(f'{seed}\n' for seed in seeds))
        # Older official generators only accept seeds.txt and write ./in.
        with tempfile.TemporaryDirectory(prefix='ale-generate-') as temp:
            subprocess.run([str(directory / 'tools/target/release/gen'), str(seedfile)], cwd=temp, check=True)
            shutil.copytree(Path(temp) / 'in', directory / split, dirs_exist_ok=True)
        if len(list((directory / split).glob('*.txt'))) != len(seeds):
            raise RuntimeError(f'{problem}: missing generated {split} cases')
    dump(directory / 'ready.json', {'archive_sha256': sha(archive), 'metadata_sha256': sha(directory / 'data.json'),
                                   'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                                   'rust': subprocess.check_output(['cargo', '--version'], text=True).strip()})


def run(command, directory, timeout, input_path=None, stem='process'):
    out_path, err_path = directory / f'{stem}.stdout', directory / f'{stem}.stderr'
    with open(input_path or os.devnull, 'rb') as inp, out_path.open('wb') as out, err_path.open('wb') as err:
        p = subprocess.Popen(command, cwd=directory, stdin=inp, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -signal.SIGKILL
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
    return rc, out_path, err_path


def case_eval(case, binary, tester, metadata):
    from ale_bench.result import CaseResult, JudgeResult
    started = time.monotonic()
    limit = metadata['constraints']['time_limit']
    memory = metadata['constraints']['memory_limit']
    with tempfile.TemporaryDirectory(prefix='ale-case-') as temp:
        work = Path(temp)
        profile = work / 'time.txt'
        candidate = ['/usr/bin/time', '-f', '%U %S %M %x', '-o', str(profile),
                     'prlimit', f'--as={memory}', f'--cpu={math.ceil(limit)}', '--fsize=16777216', '--', str(binary)]
        interactive = metadata['metadata']['problem_type'] == 'reactive'
        command = [str(tester), *candidate] if interactive else candidate
        command = ['prlimit', '--fsize=16777216', '--', *command]
        rc, output, error = run(command, work, timeout=limit * 2 + 3, input_path=case)
        cpu = 0.0
        rss = 0
        execution_rc = rc
        if profile.exists():
            try:
                user, system, mem, exit_status = profile.read_text().strip().splitlines()[-1].split()
                cpu, rss, execution_rc = float(user) + float(system), int(mem) * 1024, int(exit_status)
            except (ValueError, IndexError):
                pass
        message = error.read_text(errors='replace')[-4000:]
        status = JudgeResult.ACCEPTED
        score = 0
        if cpu > limit or rc == -signal.SIGKILL or execution_rc in (137, 152):
            status = JudgeResult.TIME_LIMIT_EXCEEDED
        elif execution_rc or (rc and not interactive):
            status = JudgeResult.RUNTIME_ERROR
        else:
            if not interactive:
                rc, _, error = run([str(tester), str(case), str(output)], work, 30, stem='judge')
                message = error.read_text(errors='replace')[-4000:]
            found = re.findall(r'^Score\s*=\s*(\d+)\s*$', message, re.M)
            if rc or not found:
                status = JudgeResult.WRONG_ANSWER
            else:
                score = int(found[-1])
        return CaseResult(judge_result=status, message=message, error_str=message,
                          absolute_score=score, execution_time=cpu or time.monotonic() - started, memory_usage=rss)


def evaluate_program(program, problem, data_root=DEFAULT_DATA, split='public', count=None, workers=1, progress=False):
    from ale_bench import constants
    from ale_bench.result import Result, ResourceUsage, CaseResult, JudgeResult
    directory = Path(data_root).resolve() / problem
    if not (directory / 'ready.json').exists():
        raise RuntimeError(f'Run prepare first: {directory}')
    metadata = json.loads((directory / 'data.json').read_text())
    seeds = metadata['seeds'][split]
    if count is not None and not 1 <= count <= len(seeds):
        raise ValueError('Invalid case count')
    cases = [directory / split / f'{i:04d}.txt' for i in range(count or len(seeds))]
    allowed = getattr(constants, f'ALLOW_SCORE_NON_AC_{split.upper()}')
    with tempfile.TemporaryDirectory(prefix='ale-program-') as temp:
        work = Path(temp)
        code = Path(program).read_text().replace('# EVOLVE-BLOCK-START', '').replace('# EVOLVE-BLOCK-END', '')
        source = work / 'main.cpp'
        source.write_text(code)
        binary = work / 'main'
        compile_command = [cxx_compiler(), '-std=c++20', '-O2', '-pipe', str(source), '-o', str(binary)]
        rc, _, error = run(compile_command, work, 90, stem='compile')
        # GCC 9 implements the C++20 draft under the older ``c++2a`` spelling.
        # Retry only when the compiler rejects the standard flag itself, so real
        # candidate compilation errors are reported without a redundant build.
        compile_error = error.read_text(errors='replace')
        if rc and 'unrecognized command line option' in compile_error and '-std=c++20' in compile_error:
            compile_command[1] = '-std=c++2a'
            rc, _, error = run(compile_command, work, 90, stem='compile')
        if rc:
            message = error.read_text(errors='replace')[-4000:]
            results = [CaseResult(judge_result=JudgeResult.COMPILATION_ERROR, message=message, error_str=message,
                                  absolute_score=0, execution_time=0, memory_usage=0) for _ in cases]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = []
                for index, result in enumerate(pool.map(lambda case: case_eval(case, binary, directory / 'tools/target/release/tester', metadata), cases), 1):
                    results.append(result)
                    if progress and (index % 25 == 0 or index == len(cases)):
                        print(f'{problem} {split}: {index}/{len(cases)} cases evaluated', flush=True)
    return Result(allow_score_non_ac=problem in allowed, resource_usage=ResourceUsage(), case_results=results)


def performance(result, problem, data_root=DEFAULT_DATA):
    """Official full-private-set ranking, including relative-score reranking."""
    from ale_bench.data import Standings, RelativeResults, RankPerformanceMap
    directory = Path(data_root) / problem
    metadata = json.loads((directory / 'data.json').read_text())
    if len(result.case_results) != len(metadata['seeds']['private']):
        raise ValueError('Private performance requires ALL private cases, never a subset')
    def rows(name):
        with (directory / name).open() as f:
            return list(csv.DictReader(f))
    relative = None
    info = metadata['metadata']
    if 'relative_score_type' in info:
        entries = rows('relative_results.csv')
        columns = [f'private_{i}' for i in range(len(result.case_results))]
        if list(entries[0]) != columns:
            raise ValueError('Private relative-score columns mismatch')
        relative = RelativeResults(absolute_scores=[[int(row[key]) for row in entries] for key in columns],
                                   relative_score_type=info['relative_score_type'], relative_max_score=info['relative_max_score'])
    standings = Standings(standings_scores=[(int(r['rank']), int(r['score'])) for r in rows('standings_scores.csv')], relative_results=relative)
    mapping = RankPerformanceMap(raw_data=[(int(r['rank']), int(r['performance'])) for r in rows('performance.csv')])
    rank, performance_rank, _ = standings.get_new_rank(result)
    return {'private_rank': rank, 'private_performance': mapping.get_performance(performance_rank),
            'private_score': result.overall_absolute_score,
            'passed': sum(c.judge_result.value == 'ACCEPTED' for c in result.case_results), 'cases': len(result.case_results)}


def public_metrics(result, score_type):
    """Positive higher-is-better fitness compatible with EE's zero curve floor."""
    from ale_bench.result import JudgeResult
    valid = result.overall_judge_result == JudgeResult.ACCEPTED
    raw = result.overall_absolute_score / len(result.case_results)
    score = raw if score_type == 'maximize' else (1e9 / (1 + raw) if valid else 0.0)
    metrics = {'combined_score': score, 'raw_mean_score': raw, 'judge_result': result.overall_judge_result.value,
               'num_cases': len(result.case_results), 'passed_cases': sum(c.judge_result == JudgeResult.ACCEPTED for c in result.case_results)}
    # Partial public scores are permitted by some official tasks. Preserve them.
    if not valid and (not result.allow_score_non_ac or score_type == 'minimize'):
        metrics.update(validity=0.0, error=next(c.message for c in result.case_results if c.judge_result != JudgeResult.ACCEPTED))
    return metrics


def evaluate(program_path):
    problem = os.environ['ALE_PROBLEM']
    data = Path(os.environ.get('ALE_SUITE_DATA', str(DEFAULT_DATA)))
    result = evaluate_program(program_path, problem, data, count=int(os.environ.get('ALE_PUBLIC_CASES', '50')),
                              workers=int(os.environ.get('ALE_WORKERS', '1')))
    metadata = json.loads((data / problem / 'data.json').read_text())
    return public_metrics(result, metadata['metadata']['score_type'])


if __name__ == "__main__":
    import argparse
    import json
    import math
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    result = evaluate(args.program_path)

    combined_score = float(result.get("combined_score", 0.0) or 0.0)
    explicit_correct = result.get("correct")
    if isinstance(explicit_correct, dict):
        explicit_correct = explicit_correct.get("correct")
    validity = result.get("validity")
    runs_successfully = result.get("runs_successfully")
    if explicit_correct is not None:
        is_correct = bool(explicit_correct)
    elif validity is not None:
        is_correct = bool(validity) and math.isfinite(combined_score)
    elif runs_successfully is not None:
        is_correct = bool(runs_successfully) and math.isfinite(combined_score)
    else:
        is_correct = not result.get("error") and math.isfinite(combined_score)

    metrics = {
        "combined_score": combined_score,
        "public": result,
        "private": {},
        "text_feedback": str(result.get("text_feedback", result.get("error", ""))),
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as output:
        json.dump(metrics, output)
    with open(os.path.join(args.results_dir, "correct.json"), "w") as output:
        json.dump({"correct": is_correct}, output)
