from threading import Condition


def start(renderer, *args, **kwargs):
    renderer.output = ""
    renderer.condition = Condition()
    while True:
        if renderer.output is None:
            renderer.condition.acquire()
            command, args, kwargs = renderer.render_command
            retval = renderer.__getattribute__(command)(*args, **kwargs)
            if retval is not None:
                renderer.output = retval
            else:
                renderer.output = ""
            renderer.condition.release()
