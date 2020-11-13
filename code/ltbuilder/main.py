import enum

from typing import Tuple


class CIRCUIT_COMPONENTS(enum.Enum):
    RESISTOR = 0
    INDUCTOR = 1
    CAPACITOR = 2
    SOURCE = 3
    DIODE = 4
    GROUND = 5


class LTSpiceUnit:
    def __init__(self, value):
        # our coordinate system considers values in 1 steps. The ltspice coordinate
        # system is based with a base of 16 => only use multiples of 16 in ltspice
        self.value = value

    def __add__(self, other):
        return LTSpiceUnit(self.value + other.value)

    def __sub__(self, other):
        return LTSpiceUnit(self.value - other.value)

    def ltspice(self):
        return self.value * 16

    @property
    def unit(self):
        return 16


class Component:
    def __init__(
        self,
        name,
        x,
        y,
        rotation,
        symbol_name,
        xoffset,
        yoffset,
        width,
        height,
        component_type,
    ):
        self.name = name
        self.x = LTSpiceUnit(x)
        self.y = LTSpiceUnit(y)
        self.rotation = rotation

        self.symbol_name = symbol_name
        self.xoffset = LTSpiceUnit(xoffset)
        self.yoffset = LTSpiceUnit(yoffset)
        self.width = LTSpiceUnit(width)
        self.height = LTSpiceUnit(height)
        self.type = component_type

    @property
    def symbol(self):
        nx, ny = self.normalize()

        return (
            f"SYMBOL {self.symbol_name} {nx.ltspice()} {ny.ltspice()} R{self.rotation}"
        )

    def normalize(self):
        # TODO generalize
        if self.rotation == 0:
            return self.x - self.xoffset, self.y - self.yoffset
        elif self.rotation == 90:
            return self.x + self.yoffset, self.y - self.xoffset
        elif self.rotation == 180:
            return self.x + self.xoffset, self.y + self.yoffset
        elif self.rotation == 270:
            return self.x - self.yoffset, self.y + self.xoffset

        raise ValueError(
            f"Rotation {self.rotation} not allowed. (Allowed: 0, 90, 180, 270)"
        )

    @property
    def start(self):
        return (self.x.value, self.y.value)

    @property
    def end(self):
        # TODO generalize
        if self.rotation == 0:
            _end = self.x, self.y + self.height
        elif self.rotation == 90:
            _end = self.x - self.height, self.y
        elif self.rotation == 180:
            _end = self.x, self.y - self.height
        elif self.rotation == 270:
            _end = self.x + self.height, self.y
        else:
            _end = (None, None)

        x, y = _end

        return x.value, y.value

    def write(self, fd):
        print(self.symbol, file=fd)

class Resistor(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "res", 1, 1, 3, 5, CIRCUIT_COMPONENTS.RESISTOR
        )


class Inductor(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "ind", 1, 1, 3, 5, CIRCUIT_COMPONENTS.INDUCTOR
        )


class Capacitor(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "cap", 1, 0, 3, 4, CIRCUIT_COMPONENTS.CAPACITOR
        )


class Source(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "voltage", 0, 1, 5, 5, CIRCUIT_COMPONENTS.SOURCE
        )


class Diode(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "diode", 1, 0, 3, 4, CIRCUIT_COMPONENTS.DIODE
        )


class Wire:
    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        x1, y1 = p1
        x2, y2 = p2
        self.x1, self.y1 = LTSpiceUnit(x1), LTSpiceUnit(y1)
        self.x2, self.y2 = LTSpiceUnit(x2), LTSpiceUnit(y2)

    @property
    def symbol(self):
        return "WIRE {x1} {y1} {x2} {y2}".format(
            x1=self.x1.ltspice(),
            y1=self.y1.ltspice(),
            x2=self.x2.ltspice(),
            y2=self.y2.ltspice(),
        )

    def write(self, fd):
        print(self.symbol, file=fd)

# TODO
# class Ground(Component):
#     super().__init__()


with open("ltspice_circuits/g.asc", "w") as f:
    print("VERSION 4", file=f)
    print("SHEET 1 880 680", file=f)
    # print(Resistor("bla", 3, 3, 0).symbol, file=f)
    # print(Resistor("bla", 3, 3, 90).symbol, file=f)
    # print(Resistor("bla", 3, 3, 180).symbol, file=f)
    # print(Resistor("bla", 3, 3, 270).symbol, file=f)

    # print(Diode("bla", 3, 3, 0).symbol, file=f)
    # print(Diode("bla", 3, 3, 90).symbol, file=f)
    # print(Diode("bla", 3, 3, 180).symbol, file=f)
    # print(Diode("bla", 3, 3, 270).symbol, file=f)

    # print(Source("bla", 3, 3, 0).symbol, file=f)
    # print(Source("bla", 3, 3, 90).symbol, file=f)
    # print(Source("bla", 3, 3, 180).symbol, file=f)
    # print(Source("bla", 3, 3, 270).symbol, file=f)

    r1 = Resistor("bla", 3, 3, 90)
    r1.write(f)

    d1 = Diode("bla", 10, 3, 90)
    d1.write(f)

    Wire(r1.start, d1.end).write(f)

    s1 = Source("bla", 18, 3, 90)
    s1.write(f)

    Wire(d1.start, s1.end).write(f)
