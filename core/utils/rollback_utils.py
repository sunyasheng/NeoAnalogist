def need_rollback(step: int):
    if step % 3 == 0:
        return True
    else:
        return False
