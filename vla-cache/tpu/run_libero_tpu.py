"""LIBERO closed-loop eval of OpenVLA + VLA-Cache on one TPU chip.

Spawns --envs LIBERO env workers (LIBERO venv) and serves them round-robin from one
device model; each episode keeps its own KV buffer and VLA-Cache state. Per-step latency
is measured on the policy call only, so interleaving does not distort it.

  python run_libero_tpu.py --mode static --chip 0 --tasks 0-9 --trials 50 --envs 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing.connection import Client

HERE = os.path.dirname(os.path.abspath(__file__))
LIBERO_PY = os.path.expanduser("~/venvs/libero/bin/python")


def pin_chip(chip):
    # one process per TPU v4 chip
    os.environ.update(TPU_VISIBLE_CHIPS=str(chip), TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
                      TPU_PROCESS_BOUNDS="1,1,1", TPU_PROCESS_PORT=str(8476 + chip),
                      TPU_PROCESS_ADDRESSES=f"localhost:{8476 + chip}", CLOUD_TPU_TASK_ID="0")


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class EnvWorker:
    def __init__(self, port, suite):
        env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", LP_NUM_THREADS="8")
        for k in [k for k in env if k.startswith("TPU_") or k == "PJRT_DEVICE"]:
            env.pop(k)
        self.proc = subprocess.Popen([LIBERO_PY, os.path.join(HERE, "libero_env_worker.py"), "--port", str(port),
                                      "--suite", suite], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                self.conn = Client(("localhost", port), authkey=b"jikji")
                break
            except ConnectionRefusedError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"env worker on port {port} did not start")

    def call(self, **msg):
        self.conn.send(msg)
        return self.conn.recv()

    def close(self):
        try:
            self.conn.send(dict(cmd="close"))
        except Exception:
            pass
        self.proc.wait(timeout=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/models/openvla-7b-finetuned-libero-spatial"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--mode", default="static", choices=["off", "static", "dynamic"])
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--trials", default="0-49", help="episode (init state) indices")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--record-dir", default=None, help="save frames + exact VLA-Cache plans per episode")
    args = ap.parse_args()

    pin_chip(args.chip)
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    from PIL import Image

    sys.path.insert(0, HERE)
    from check_equivalence import load
    from policy import VLACachePolicy

    out_path = args.out or os.path.join(HERE, "results", f"libero_{args.suite}_{args.mode}_chip{args.chip}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_eps = set()
    if os.path.exists(out_path):  # resume
        for line in open(out_path):
            r = json.loads(line)
            if r.get("type") == "episode":
                done_eps.add((r["task_id"], r["episode"]))
    log = open(out_path, "a")

    model, proc = load(args.ckpt, torch.bfloat16)
    policy = VLACachePolicy(model, proc, xm.xla_device(), mode=args.mode, bucket=args.bucket, unnorm_key=args.suite,
                            record_plan=args.record_dir is not None)
    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)

    queue = [(t, e) for t in parse_range(args.tasks) for e in parse_range(args.trials) if (t, e) not in done_eps]
    workers = [EnvWorker(7000 + 100 * args.chip + i, args.suite) for i in range(min(args.envs, len(queue)))]
    max_steps = workers[0].call(cmd="info")["max_steps"] if workers else 0
    print(f"chip {args.chip} mode {args.mode}: {len(queue)} episodes, {len(workers)} envs", flush=True)

    def start(w):
        task_id, ep = queue.pop(0)
        r = w.call(cmd="reset", task_id=task_id, episode=ep)
        return dict(task_id=task_id, episode=ep, task=r["task"], image=r["image"], t=0, state=None, steps=[],
                    pending=False, frames=[], plans=[])

    def act(a, w):
        """Policy step for one episode; the env step is sent without waiting so the simulator
        runs while the device serves the other episodes."""
        img = Image.fromarray(np.frombuffer(a["image"], np.uint8).reshape(224, 224, 3))
        policy.set_state(a["state"])
        action, _, info = policy.step(img, a["task"])
        a["state"] = policy.get_state()
        # original post-processing: gripper [0,1] -> [-1,1], binarize, invert for OpenVLA
        action = action.copy()
        action[-1] = -np.sign(2 * action[-1] - 1)
        if args.record_dir:
            a["frames"].append(np.frombuffer(a["image"], np.uint8).reshape(224, 224, 3))
            a["plans"].append(dict(info.get("plan", {}), prompt_len=info["prompt_len"]))
        tm = info["timing"]
        a["steps"].append([round(tm["prefill"] * 1e3, 2), round(tm["decode"] * 1e3, 2), round(tm["host_plan"] * 1e3, 2),
                           info.get("rows"), info.get("n_reusable")])
        w.conn.send(dict(cmd="step", action=action.tolist()))
        a["pending"] = True

    active = {i: start(w) for i, w in enumerate(workers)}
    n_done = n_succ = 0
    t_start = time.time()
    while active:
        for i in list(active):
            a, w = active[i], workers[i]
            if not a["pending"]:
                act(a, w)
                continue
            r = w.conn.recv()
            a["pending"] = False
            a["t"] += 1
            finished = r["done"] or r["error"] is not None or a["t"] >= max_steps
            if not finished:
                a["image"] = r["image"]
                act(a, w)
                continue
            n_done += 1
            n_succ += int(r["done"])
            log.write(json.dumps(dict(type="episode", mode=args.mode, bucket=args.bucket, task_id=a["task_id"],
                                      episode=a["episode"], task=a["task"], success=bool(r["done"]), steps=a["t"],
                                      error=r["error"], per_step=a["steps"])) + "\n")
            log.flush()
            if args.record_dir:
                np.savez_compressed(os.path.join(args.record_dir, f"task{a['task_id']:02d}_ep{a['episode']:02d}.npz"),
                                    frames=np.stack(a["frames"]), task=a["task"], task_id=a["task_id"],
                                    episode=a["episode"], success=bool(r["done"]), plans=json.dumps(a["plans"]))
            el = time.time() - t_start
            print(f"[chip {args.chip} {args.mode}] task {a['task_id']} ep {a['episode']}: "
                  f"{'success' if r['done'] else 'fail'} in {a['t']} steps | {n_succ}/{n_done} "
                  f"({100 * n_succ / n_done:.1f}%) | {el / 60:.1f} min", flush=True)
            if queue:
                active[i] = start(w)
            else:
                del active[i]
    for w in workers:
        w.close()


if __name__ == "__main__":
    main()
