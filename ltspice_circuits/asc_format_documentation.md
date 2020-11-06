# .asc fileformat reverseing

## Syntax

Version 4
SHEET <Number> xlen ylen
WIRE xstart ystart xend yend
SYMBOL <res/voltage/cap/ind> x y <rotation(RXX)>
SYMATTR <InstName/Value> <Name/Value> // symattr always applies on previous element

## Offsets

Coordinate system starts at upper left corner

### Vertical connections

    // used schematic
    Version 4
    SHEET 1 880 680
    WIRE 100 100 100 150
    WIRE 100 150 100 250
    WIRE 100 250 100 350
    WIRE 100 350 100 550
    WIRE 100 550 100 650
    SYMBOL res 84 134 R0
    SYMBOL cap 84 250 R0
    SYMBOL ind 84 334 R0
    SYMBOL voltage 100 534 R0

    // vertical wire
    WIRE 100 100 100 150
    // connect resistor at the bottom of the wire
    SYMBOL res 84 134 R0
    // xwire - xres = 16 => R180 xwire + 16 = xres
    // ywire - yres = 16 => R180 ywire - 34 = yres
    // connect a wire at the bottom of the resistor
    WIRE 100 150 100 250
    // the wire perfectly matches when it's connected to the previous wire
    // the wire does not connect? when y2 is <= 230 (weird) e.g.
    WIRE 100 150 100 230 // does not connect?
    // connect capacitor to the bottom of the wire 
    SYMBOL cap 84 250 R0
    // xwire - xcap = 16
    // ywire - ycap = 0
    WIRE 100 250 100 350
    // wire can also be connected with the previous values
    // connect inductor
    SYMBOL ind 84 334 R0
    // xwire - xind = 16
    // ywire - yind = 16
    WIRE 100 350 100 550
    // connect voltage
    SYMBOL voltage 100 534 R0
    // xwire - xvolt = 0
    // ywire - yvolt = 16

### Horizontal connections

    WIRE 100 100 150 100
    SYMBOL res 246 84 R90 
    // xwire + 96 = xres
    // ywire - 16 = yres


### Offsets 2

### Legend
    
    W = wire
    C = capacitor
    S = source
    I = inductor
    D = diode
    G = ground
    R = resistor
    t = top
    b = bottom
    r = right
    l = left
    h = height
    w = width


### Capacitor

    R0: (vertical)
      yCt = yWt
      xC  = xW - 16
      hC  = 64

    R90:  
      yCt = yWt - 16
      xCl = xWl 
      wC  = 64
      yCb = yWt + 64
      
    R180: 
      yCb = yWb
      xC  = xW + 16
      yCt = yWb - 64

    R270: 
      
      






