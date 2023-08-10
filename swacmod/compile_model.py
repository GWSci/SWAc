from swacmod import utils as u

boo = u.compile_model()


def get_status():
    return 'model build: ' + str(boo)
