from threading import Condition


class RendererController:
    running = True
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RendererController, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def start(self, renderer):
        renderer.output = ""
        renderer.condition = Condition()
        renderer.output_for_display = True
        while self.running:
            if renderer.output is None:
                renderer.condition.acquire()
                command, args, kwargs = renderer.render_command
                retval = renderer.__getattribute__(command)(*args, **kwargs)
                if retval is not None:
                    renderer.output = retval
                else:
                    renderer.output = ""
                renderer.condition.notify()
                renderer.condition.release()

        renderer.context.pop()
        renderer.context = None

        from pycuda.tools import clear_context_caches
        clear_context_caches()

    def stop(self):
        self.running = False
