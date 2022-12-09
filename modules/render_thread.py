import threading
import time

import modules.call_queue as call_queue

from modules import shared
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images


class StateQueue:
    def __init__(self):
        self.next_index = 0
        self.pending = []
        self.finished = []

    def enqueue(self, batch_kind, info_html, args):
        self.pending.append(BatchJob(self.next_index, batch_kind, info_html, args))
        self.next_index += 1
        return len(self.pending), len(self.finished)

    def delete(self, queue_name, queue_index, job_index):
        if queue_name == "pending":
            ls = self.pending
        else:
            ls = self.finished

        if queue_index < 0 or len(ls) >= queue_index:
            return None

        job = ls[queue_index]
        if job.index != job_index:
            return None

        return ls.pop(queue_index)


class BatchJob:
    def __init__(self, index, batch_kind, info_html, args):
        self.batch_kind = batch_kind
        self.index = index
        self.status = "pending"
        self.args = args
        self.images = []
        self.processed_js = {}
        self.info_html = info_html


LOCK_TIMEOUT = 15

state_queue = StateQueue()
thread = None
thread_alive = False

lock = threading.RLock()


def queue_check():
    if call_queue.queue_lock.locked():
        return

    with call_queue.queue_lock:
        if len(state_queue.pending) == 0:
            return

        item = state_queue.pending.pop(0)

        print("+++++++++ start +++++++++")

        images = []
        processed_js = {}
        info_html = ""

        try:
            import modules.img2img
            import modules.txt2img
            import modules.extras

            if item.batch_kind == "txt2img":
                images, processed_js, info_html = modules.txt2img.txt2img(*item.args)
            elif item.batch_kind == "img2img":
                images, processed_js, info_html = modules.img2img.img2img(*item.args)
            elif item.batch_kind == "extras":
                images, info_html, _ = modules.extras.run_extras(*item.args)
            else:
                print(f"unknown batch kind {item.batch_kind}!")

            item.images = images
            item.processed_js = processed_js
            item.info_html = info_html
            item.status = "succeeded"
        except Exception as e:
            print("Error processing: " + str(e))
            item.info_html = f"<div>Error: {str(e)}</div>"
            item.status = "failed"

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        state_queue.finished.append(item)

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
