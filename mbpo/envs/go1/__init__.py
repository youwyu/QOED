"""Go1 MuJoCo environments."""

__all__ = ["Go1JoystickEnv"]


def __getattr__(name):
    if name == "Go1JoystickEnv":
        from .joystick import Go1JoystickEnv

        return Go1JoystickEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
