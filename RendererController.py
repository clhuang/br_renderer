from threading import _Condition, Lock


class CCondition(_Condition):
    x = 0

    def acquire(self, *args):
        self.x += 1
        print self.x
        super(CCondition, self).acquire(*args)

    def release(self, *args):
        self.x -= 1
        print self.x
        super(CCondition, self).release(*args)


class RendererController:
    running = True
    render_command = None

    def __init__(self, renderer):
        self.rend = renderer

    def start(self):
        self.rend.asdf = 0
        self.output = ""
        self.condition = CCondition()

        self.output_for_display = True
        while self.running:
            if self.output is None:
                self.condition.acquire()
                print 'contacquire'
                command, args, kwargs = self.render_command
                retval = self.rend.__getattribute__(command)(*args, **kwargs)
                if retval is not None:
                    self.output = retval
                else:
                    self.output = ""
                self.condition.notify()
                self.condition.release()
                print 'contrelease'

        self.rend.context.pop()
        self.rend.context = None

        from pycuda.tools import clear_context_caches
        clear_context_caches()

    def stop(self):
        self.running = False
