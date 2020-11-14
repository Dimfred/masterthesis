from . import *

if __name__ == "__main__":

    r1 = Resistor("bla", 3, 3, 90)
    d1 = Diode("bla", 10, 3, 90)
    w1 = Wire(r1.start, d1.end)
    s1 = Source("bla", 18, 3, 90)
    w2 = Wire(d1.start, s1.end)

    writer = LTWriter()
    writer.write("ltspice_circuits/g.asc", [r1, d1, w1, s1, w2])
