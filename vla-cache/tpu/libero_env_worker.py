"""LIBERO env worker (runs in the LIBERO venv). Talks to the policy process over a
multiprocessing connection using plain Python types only (the two venvs have different numpy).

Image preprocessing is the original eval's: rotate 180, JPEG round-trip, lanczos3 resize to 224
(libero_utils.get_libero_image), then the 0.9-area center crop (openvla_utils.process_image).

  MUJOCO_GL=egl ~/venvs/libero/bin/python libero_env_worker.py --port 6000 --suite libero_spatial
"""

import argparse
import os
from multiprocessing.connection import Listener

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

tf.config.set_visible_devices([], "GPU")
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
NUM_STEPS_WAIT = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def preprocess(agentview, resize=224, crop_scale=0.9):
    img = agentview[::-1, ::-1]
    img = tf.io.decode_image(tf.image.encode_jpeg(img), expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (resize, resize), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    # center crop (process_image / crop_and_resize)
    x = tf.image.convert_image_dtype(img, tf.float32)[None]
    s = float(np.clip(np.sqrt(crop_scale), 0, 1))
    off = (1 - s) / 2
    x = tf.image.crop_and_resize(x, [[off, off, off + s, off + s]], [0], (224, 224))[0]
    x = tf.image.convert_image_dtype(tf.clip_by_value(x, 0, 1), tf.uint8, saturate=True)
    return np.ascontiguousarray(x.numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    env, env_task, t, obs = None, None, 0, None
    with Listener(("localhost", args.port), authkey=b"jikji") as listener, listener.accept() as conn:
        while True:
            msg = conn.recv()
            cmd = msg["cmd"]
            if cmd == "close":
                break
            if cmd == "info":
                conn.send(dict(n_tasks=suite.n_tasks, max_steps=MAX_STEPS[args.suite]))
            elif cmd == "reset":
                task_id, ep = msg["task_id"], msg["episode"]
                task = suite.get_task(task_id)
                if env_task != task_id:
                    if env is not None:
                        env.close()
                    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                    env.seed(0)  # as the original eval: seed affects object placement
                    env_task = task_id
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[ep])
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                conn.send(dict(image=preprocess(obs["agentview_image"]).tobytes(), task=task.language))
            elif cmd == "step":
                try:
                    obs, _, done, _ = env.step(list(msg["action"]))
                    conn.send(dict(image=preprocess(obs["agentview_image"]).tobytes(), done=bool(done), error=None))
                except Exception as e:  # the original eval ends the episode on any exception
                    conn.send(dict(image=None, done=False, error=repr(e)))


if __name__ == "__main__":
    main()
