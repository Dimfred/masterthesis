import enum

from typing import Tuple, List


class CIRCUIT_COMPONENTS(enum.Enum):
    RESISTOR = 0
    INDUCTOR = 1
    CAPACITOR = 2
    DIODE = 3
    SOURCE = 4
    CURRENT = 5
    GROUND = 6


class CONNECTION_ORIENTATION(enum.Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


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
        return self.x.value, self.y.value

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

    @property
    def connections(self):
        # returns either top, bottom OR left, right
        # depending on the orientation of the component
        if self.rotation == 0 or self.rotation == 270:
            return self.start, self.end
        # TODO could be else but maybe other rotations
        elif self.rotation == 90 or self.rotation == 180:
            return self.end, self.start

    @property
    def left(self):
        return self.connections[0]

    @property
    def right(self):
        return self.connections[1]

    @property
    def top(self):
        return self.connections[0]

    @property
    def bottom(self):
        return self.connections[1]

    def write(self, fd):
        print(self.symbol, file=fd)

    @property
    def is_horizontal(self):
        return self.rotation == 90 or self.rotation == 270

    @property
    def is_vertical(self):
        return not self.is_horizontal


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


class Diode(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "diode", 1, 0, 3, 4, CIRCUIT_COMPONENTS.DIODE
        )


class Source(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "voltage", 0, 1, 5, 5, CIRCUIT_COMPONENTS.SOURCE
        )


class Current(Component):
    def __init__(self, name, x, y, rotation):
        super().__init__(
            name, x, y, rotation, "current", 0, 0, 4, 5, CIRCUIT_COMPONENTS.CURRENT
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


class Ground:
    def __init__(self, name, x, y, *args):
        self.x, self.y = LTSpiceUnit(x), LTSpiceUnit(y)

    @property
    def symbol(self):
        return "FLAG {x} {y} 0".format(x=self.x.ltspice(), y=self.y.ltspice())

    def write(self, fd):
        print(self.symbol, file=fd)

    @property
    def connections(self):
        # returns either top, bottom OR left, right
        # depending on the orientation of the component
        if self.rotation == 0 or self.rotation == 270:
            return self.start, self.end
        # TODO could be else but maybe other rotations
        elif self.rotation == 90 or self.rotation == 180:
            return self.end, self.start

    @property
    def start(self):
        return self.x.value, self.y.value

    @property
    def end(self):
        return self.start

    @property
    def left(self):
        return self.start

    @property
    def right(self):
        return self.left

    @property
    def top(self):
        return self.left

    @property
    def bottom(self):
        return self.left

    def write(self, fd):
        print(self.symbol, file=fd)

    @property
    def is_horizontal(self):
        return False

    @property
    def is_vertical(self):
        return True



class LTWriter:
    def __init__(self, version=4, sheet=(880, 680)):
        self.version = version
        self.sheet = sheet

    def write(self, file, components: List[Component]):
        with open(file, "w") as f:
            print(f"VERSION {self.version}", file=f)
            x, y = self.sheet
            # TODO maybe change the one?
            print(f"SHEET 1 {x} {y}", file=f)

            for component in components:
                component.write(f)
