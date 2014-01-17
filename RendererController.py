from threading import Condition


class RendererController:
    running = True
    render_command = None

    def __init__(self, renderer):
        self.rend = renderer

    def start(self):
        self.output = ""
        self.condition = Condition()
        self.output_for_display = True
        while self.running:
            if self.output is None:
                self.condition.acquire()
                command, args, kwargs = self.render_command
                retval = self.rend.__getattribute__(command)(*args, **kwargs)
                if retval is not None:
                    self.output = retval
                else:
                    self.output = ""
                self.condition.notify()
                self.condition.release()

        self.rend.context.pop()
        self.rend.context = None

        from pycuda.tools import clear_context_caches
        clear_context_caches()

    def stop(self):
        self.running = False
