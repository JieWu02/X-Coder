import threading
import os


_orig_threading_excepthook = threading.excepthook


def _threading_excepthook(args):
    _orig_threading_excepthook(args)
    print(f"Exiting process due to unhandled exception in thread {args.thread.name}")
    os._exit(1)


def install_thread_excepthook():
    threading.excepthook = _threading_excepthook


def uninstall_thread_excepthook():
    threading.excepthook = _orig_threading_excepthook
