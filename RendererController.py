from threading import Condition

running = True
rend = None


def start(renderer):
    global rend
    rend = renderer
    renderer.output = ""
    renderer.condition = Condition()
    renderer.output_for_display = True
    i = 0
    while running:
        print str(running) + ' ' + str(i)
        i += 1
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


def cleanup():
    global running
    running = False
