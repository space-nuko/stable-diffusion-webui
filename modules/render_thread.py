import threading
import time

import modules.call_queue as call_queue

from modules import shared
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images


class StateQueue:
    def __init__(self):
        self.pending = []
        self.finished = []

LOCK_TIMEOUT = 15

state_queue = StateQueue()
thread = None
thread_alive = False

lock = threading.RLock()


def queue_check():
    if call_queue.queue_lock.locked():
        print("already queued")
        return

    with call_queue.queue_lock:
        if len(state_queue.pending) == 0:
            return

        item = state_queue.pending.pop(0)

        print("+++++++++ start +++++++++")

        p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt="1girl",
            styles=[],
            negative_prompt="",
            seed=-1,
            subseed=-1,
            #subseed_strength=subseed_strength,
            #seed_resize_from_h=seed_resize_from_h,
            #seed_resize_from_w=seed_resize_from_w,
            #seed_enable_extras=seed_enable_extras,
            sampler_name="DDIM",
            batch_size=4,
            n_iter=1,
            steps=30,
            cfg_scale=0.7,
            width=512,
            height=512,
            restore_faces=False,
            tiling=False,
            enable_hr=False,
            denoising_strength=None,
            firstphase_width=None,
            firstphase_height=None,
        )

        shared.state.begin()

        processed = process_images(p)
        state_queue.finished.append(processed)

        shared.state.end()

        print("finish")


def render_thread():
    global thread_alive

    thread_alive = True

    while thread_alive:
        queue_check()
        time.sleep(1.0)


def stop_render_thread(t):
    global thread, thread_alive

    if not lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception("failed to acquire render lock: stop_render_thread")
    print("Stopping batch render thread")

    thread_alive = False
    thread = None

    lock.release()


def start_render_thread():
    global thread

    if not lock.acquire(blocking=True, timeout=LOCK_TIMEOUT): raise Exception("failed to acquire render lock: start_render_thread")
    print("Starting batch render thread")

    try:
        thread = threading.Thread(target=render_thread)
        thread.daemon = True
        thread.name = "render_thread"
        thread.start()
    finally:
        lock.release()

    timeout = 15
    while not thread.is_alive():
        if timeout <= 0:
            return False
        timeout -= 1
        time.sleep(1)
    return True


def initialize():
    global thread

    if thread is not None:
        stop_render_thread()

    if not start_render_thread():
        raise Exception("Batch render thread failed to start!")
